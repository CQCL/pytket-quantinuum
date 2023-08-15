# Copyright 2020-2023 Cambridge Quantum Computing
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pytket Backend for Quantinuum devices."""

from ast import literal_eval
from base64 import b64encode
from dataclasses import dataclass
from enum import Enum
import json
from http import HTTPStatus
from typing import Dict, List, Set, Optional, Sequence, Union, Any, cast, Tuple
import warnings

import numpy as np
import requests

from pytket.backends import Backend, ResultHandle, CircuitStatus, StatusEnum
from pytket.backends.backend import KwargTypes
from pytket.backends.resulthandle import _ResultIdTuple
from pytket.architecture import FullyConnected  # type: ignore
from pytket.backends.backendinfo import BackendInfo
from pytket.backends.backendresult import BackendResult
from pytket.backends.backend_exceptions import CircuitNotRunError
from pytket.circuit import Circuit, OpType, Bit  # type: ignore
from pytket._tket.circuit import _TEMP_BIT_NAME  # type: ignore
from pytket.extensions.quantinuum._metadata import __extension_version__

try:
    from pytket.qir import pytket_to_qir, QIRFormat
    from pytket.qir import __extension_version__ as pytket_qir_version
except:
    pass
from pytket.qasm import circuit_to_qasm_str
from pytket.passes import (  # type: ignore
    BasePass,
    DecomposeTK2,
    SequencePass,
    SynthesiseTK,
    RemoveRedundancies,
    FullPeepholeOptimise,
    DecomposeBoxes,
    NormaliseTK2,
    SimplifyInitial,
    ZZPhaseToRz,
    CustomPass,
    FlattenRelabelRegistersPass,
    auto_rebase_pass,
    auto_squash_pass,
)
from pytket.predicates import (  # type: ignore
    GateSetPredicate,
    MaxNQubitsPredicate,
    MaxNClRegPredicate,
    Predicate,
    NoSymbolsPredicate,
)
from pytket.utils import prepare_circuit
from pytket.utils.outcomearray import OutcomeArray
from pytket.wasm import WasmFileHandler

from pytket.extensions.quantinuum.backends.credential_storage import (
    MemoryCredentialStorage,
)

from pytket.extensions.quantinuum.backends.leakage_gadget import get_detection_circuit
from .api_wrappers import QuantinuumAPIError, QuantinuumAPI

_DEBUG_HANDLE_PREFIX = "_MACHINE_DEBUG_"
MAX_C_REG_WIDTH = 32

_STATUS_MAP = {
    "queued": StatusEnum.QUEUED,
    "running": StatusEnum.RUNNING,
    "completed": StatusEnum.COMPLETED,
    "failed": StatusEnum.ERROR,
    "canceling": StatusEnum.CANCELLED,
    "canceled": StatusEnum.CANCELLED,
}

_GATE_SET = {
    OpType.Rz,
    OpType.PhasedX,
    OpType.ZZMax,
    OpType.Reset,
    OpType.Measure,
    OpType.Barrier,
    OpType.RangePredicate,
    OpType.MultiBit,
    OpType.ExplicitPredicate,
    OpType.ExplicitModifier,
    OpType.SetBits,
    OpType.CopyBits,
    OpType.ClassicalExpBox,
    OpType.WASM,
}


def _get_gateset(gates: List[str]) -> Set[OpType]:
    gs = _GATE_SET.copy()
    if "RZZ" in gates:
        gs.add(OpType.ZZPhase)
    if "Rxxyyzz" in gates:
        gs.add(OpType.TK2)
    return gs


def scratch_reg_resize_pass(max_size: int = MAX_C_REG_WIDTH) -> CustomPass:
    """Given a max scratch register width, return a compiler pass that
    breaks up the internal scratch bit registers into smaller registers
    """

    def trans(circ: Circuit, max_size: int = max_size) -> Circuit:
        # Find all scratch bits
        scratch_bits = [
            bit
            for bit in circ.bits
            if (
                bit.reg_name == _TEMP_BIT_NAME
                or bit.reg_name.startswith(f"{_TEMP_BIT_NAME}_")
            )
        ]
        # If the total number of scratch bits exceeds the max width, rename them
        if len(scratch_bits) > max_size:
            bits_map = {}
            for i, bit in enumerate(scratch_bits):
                bits_map[bit] = Bit(f"{_TEMP_BIT_NAME}_{i//max_size}", i % max_size)
            circ.rename_units(bits_map)
        return circ

    return CustomPass(trans, label="resize scratch bits")


class GetResultFailed(Exception):
    pass


class NoSyntaxChecker(Exception):
    pass


class MaxShotsExceeded(Exception):
    pass


class WasmUnsupported(Exception):
    pass


class BatchingUnsupported(Exception):
    """Batching not supported for this backend."""


@dataclass
class DeviceNotAvailable(Exception):
    device_name: str


class Language(Enum):
    """Language used for submission of circuits."""

    QASM = "OPENQASM 2.0"
    QIR = "QIR 1.0"


# DEFAULT_CREDENTIALS_STORAGE for use with the DEFAULT_API_HANDLER.
DEFAULT_CREDENTIALS_STORAGE = MemoryCredentialStorage()

# DEFAULT_API_HANDLER provides a global re-usable API handler
# that will persist after this module is imported.
#
# This allows users to create multiple QuantinuumBackend instances
# without requiring them to acquire new tokens.
DEFAULT_API_HANDLER = QuantinuumAPI(DEFAULT_CREDENTIALS_STORAGE)

QuumKwargTypes = Union[KwargTypes, WasmFileHandler, Dict[str, Any], OpType, bool]


class QuantinuumBackend(Backend):
    """
    Interface to a Quantinuum device.
    More information about the QuantinuumBackend can be found on this page
    https://cqcl.github.io/pytket-quantinuum/api/index.html
    """

    _supports_shots = True
    _supports_counts = True
    _supports_contextual_optimisation = True
    _persistent_handles = True

    def __init__(
        self,
        device_name: str,
        label: Optional[str] = "job",
        simulator: str = "state-vector",
        group: Optional[str] = None,
        provider: Optional[str] = None,
        machine_debug: bool = False,
        api_handler: QuantinuumAPI = DEFAULT_API_HANDLER,
        **kwargs: QuumKwargTypes,
    ):
        """Construct a new Quantinuum backend.

        :param device_name: Name of device, e.g. "H1-1"
        :type device_name: str
        :param label: Job labels used if Circuits have no name, defaults to "job"
        :type label: Optional[str], optional
        :param simulator: Only applies to simulator devices, options are
            "state-vector" or "stabilizer", defaults to "state-vector"
        :param group: string identifier of a collection of jobs, can be used for usage
          tracking.
        :type group: Optional[str], optional
        :param provider: select a provider for federated authentication. We currently
            only support 'microsoft', which enables the microsoft Device Flow.
        :type provider: Optional[str], optional
        :type simulator: str, optional
        :param api_handler: Instance of API handler, defaults to DEFAULT_API_HANDLER
        :type api_handler: QuantinuumAPI

        Supported kwargs:

        * `options`: items to add to the "options" dictionary of the request body, as a
          json-style dictionary (see :py:meth:`QuantinuumBackend.process_circuits`)
        """

        super().__init__()
        self._device_name = device_name
        self._label = label
        self._group = group

        self._backend_info: Optional[BackendInfo] = None
        self._MACHINE_DEBUG = machine_debug

        self.simulator_type = simulator

        self.api_handler = api_handler

        self.api_handler.provider = provider

        self._process_circuits_options = cast(Dict[str, Any], kwargs.get("options", {}))

    @classmethod
    def _available_devices(
        cls,
        api_handler: QuantinuumAPI,
    ) -> List[Dict[str, Any]]:
        """List devices available from Quantinuum.

        >>> QuantinuumBackend._available_devices()
        e.g. [{'name': 'H1', 'n_qubits': 6}]

        :param api_handler: Instance of API handler
        :type api_handler: QuantinuumAPI
        :return: Dictionaries of machine name and number of qubits.
        :rtype: List[Dict[str, Any]]
        """
        id_token = api_handler.login()
        if api_handler.online:
            res = requests.get(
                f"{api_handler.url}machine/?config=true",
                headers={"Authorization": id_token},
            )
            api_handler._response_check(res, "get machine list")
            jr = res.json()
        else:
            jr = api_handler._get_machine_list()  # type: ignore
        return jr  # type: ignore

    @classmethod
    def _dict_to_backendinfo(cls, dct: Dict[str, Any]) -> BackendInfo:
        name: str = dct.pop("name")
        n_qubits: int = dct.pop("n_qubits")
        n_cl_reg: Optional[int] = None
        if "n_classical_registers" in dct:
            n_cl_reg = dct.pop("n_classical_registers")
        gate_set: List[str] = dct.pop("gateset", [])
        return BackendInfo(
            name=cls.__name__,
            device_name=name,
            version=__extension_version__,
            architecture=FullyConnected(n_qubits, "q"),
            gate_set=_get_gateset(gate_set),
            n_cl_reg=n_cl_reg,
            supports_fast_feedforward=True,
            supports_midcircuit_measurement=True,
            supports_reset=True,
            misc=dct,
        )

    @classmethod
    def available_devices(
        cls,
        **kwargs: Any,
    ) -> List[BackendInfo]:
        """
        See :py:meth:`pytket.backends.Backend.available_devices`.
        :param api_handler: Instance of API handler, defaults to DEFAULT_API_HANDLER
        :type api_handler: Optional[QuantinuumAPI]
        """
        api_handler = kwargs.get("api_handler", DEFAULT_API_HANDLER)
        jr = cls._available_devices(api_handler)
        return list(map(cls._dict_to_backendinfo, jr))

    def _retrieve_backendinfo(self, machine: str) -> BackendInfo:
        jr = self._available_devices(self.api_handler)
        try:
            _machine_info = next(entry for entry in jr if entry["name"] == machine)
        except StopIteration:
            raise DeviceNotAvailable(machine)
        _machine_info["options"] = self._process_circuits_options
        return self._dict_to_backendinfo(_machine_info)

    @classmethod
    def device_state(
        cls,
        device_name: str,
        api_handler: QuantinuumAPI = DEFAULT_API_HANDLER,
    ) -> str:
        """Check the status of a device.

        >>> QuantinuumBackend.device_state('H1') # e.g. "online"


        :param device_name: Name of the device.
        :type device_name: str
        :param api_handler: Instance of API handler, defaults to DEFAULT_API_HANDLER
        :type api_handler: QuantinuumAPI
        :return: String of state, e.g. "online"
        :rtype: str
        """
        res = requests.get(
            f"{api_handler.url}machine/{device_name}",
            headers={"Authorization": api_handler.login()},
        )
        api_handler._response_check(res, "get machine status")
        return str(res.json()["state"])

    @property
    def backend_info(self) -> Optional[BackendInfo]:
        if self._backend_info is None and not self._MACHINE_DEBUG:
            self._backend_info = self._retrieve_backendinfo(self._device_name)
        return self._backend_info

    @property
    def _gate_set(self) -> Set[OpType]:
        return (
            _GATE_SET
            if self._MACHINE_DEBUG
            else cast(BackendInfo, self.backend_info).gate_set
        )

    @property
    def required_predicates(self) -> List[Predicate]:
        preds = [
            NoSymbolsPredicate(),
            GateSetPredicate(self._gate_set),
        ]
        if not self._MACHINE_DEBUG:
            assert self.backend_info is not None
            preds.append(MaxNQubitsPredicate(self.backend_info.n_nodes))
            preds.append(MaxNClRegPredicate(self.backend_info.n_cl_reg))

        return preds

    @property
    def _two_qubit_gate_set(self) -> Set[OpType]:
        """
        Assumes that only possibly supported two-qubit gates are
        ZZPhase, ZZMax and TK2.

        :return: Set of two-qubit OpType in gateset.
        :rtype: Set[OpType]
        """
        return self._gate_set & set([OpType.ZZPhase, OpType.ZZMax, OpType.TK2])

    def rebase_pass(self, **kwargs: QuumKwargTypes) -> BasePass:
        """
        Supported kwargs:
        * `implicit_swaps`: Boolean flag, which if true, returns
            rebasing pass that allows implicit wire swaps.
            Default False.
        * `target_2qb_gate`: pytket OpType, if provided, will
            return a rebasing pass that only allows given
            two-qubit gate type.
        :return: Compilation pass for rebasing circuits
        :rtype: BasePass
        """
        target_2qb_optype: OpType = kwargs.get("target_2qb_gate", OpType.ZZPhase)
        if target_2qb_optype not in self._two_qubit_gate_set:
            raise QuantinuumAPIError(
                "Requested target_2qb_gate is not supported by the given Device. "
                "Please check _two_qubit_gate_set attribute to see which two-qubit "
                "gates are supported."
            )
        return auto_rebase_pass(
            self._gate_set - self._two_qubit_gate_set | {target_2qb_optype},
            allow_swaps=bool(kwargs.get("implicit_swap", True)),
        )

    def default_compilation_pass_with_options(
        self, optimisation_level: int = 2, **kwargs: QuumKwargTypes
    ) -> BasePass:
        """
        :param optimisation_level: Allows values of 0,1 or 2, with higher values
            prompting more computationally heavy optimising compilation that
            can lead to reduced gate count in circuits.
        :type optimisation_level: int
        :return: Compilation pass for compiling circuits to Quantinuum devices
        :rtype: BasePass

        Supported kwargs:
        * `implicit_swaps`: Boolean flag, which if true, allows rebasing of
            Circuit via TK2 gates to use implicit wire swaps in circuit
            construction if it reduces the total 2qb qate count.
        * `target_2qb_gate`: pytket OpType, if provided, will rebase
            circuits such that the only two-qubit gates will be of the
            provided type.
        """
        assert optimisation_level in range(3)
        passlist = [
            DecomposeBoxes(),
            scratch_reg_resize_pass(),
        ]
        squash = auto_squash_pass({OpType.PhasedX, OpType.Rz})
        # use default (perfect fidelities) for supported gates
        fidelities: Dict[str, Any] = {}
        # If ZZPhase is available we should prefer it to ZZMax.
        if OpType.ZZPhase in self._gate_set:
            fidelities["ZZPhase_fidelity"] = lambda x: 1.0
        elif OpType.ZZMax in self._gate_set:
            fidelities["ZZMax_fidelity"] = 1.0
        else:
            raise QuantinuumAPIError(
                "Either ZZMax or ZZPhase gate must be supported by device"
            )
        # If you make changes to the default_compilation_pass,
        # then please update this page accordingly
        # https://cqcl.github.io/pytket-quantinuum/api/index.html#default-compilation
        # Edit this docs source file -> pytket-quantinuum/docs/intro.txt
        if optimisation_level == 0:
            passlist.append(self.rebase_pass(**kwargs))
        elif optimisation_level == 1:
            passlist.extend(
                [
                    SynthesiseTK(),
                    NormaliseTK2(),
                    DecomposeTK2(**fidelities),
                    self.rebase_pass(**kwargs),
                    ZZPhaseToRz(),
                    RemoveRedundancies(),
                    squash,
                    SimplifyInitial(
                        allow_classical=False, create_all_qubits=True, xcirc=_xcirc
                    ),
                ]
            )
        else:
            passlist.extend(
                [
                    FullPeepholeOptimise(target_2qb_gate=OpType.TK2),
                    NormaliseTK2(),
                    DecomposeTK2(**fidelities),
                    self.rebase_pass(**kwargs),
                    RemoveRedundancies(),
                    squash,
                    SimplifyInitial(
                        allow_classical=False, create_all_qubits=True, xcirc=_xcirc
                    ),
                ]
            )
        # In TKET, a qubit register with N qubits can have qubits
        # indexed with a a value greater than N, i.e. a single
        # qubit register can exist with index "7" or similar.
        # Similarly, a qubit register with N qubits could be defined
        # in a Circuit, but fewer than N qubits in the register
        # have operations.
        # Both of these cases can causes issues when converting to QASM,
        # as the size of the defined "qreg" can be larger than the
        # number of Qubits actually used, or at times larger than the
        # number of device Qubits, even if fewer are really used.
        # By flattening the Circuit qubit registers, we make sure
        # that the produced QASM has one "qreg", with the exact number
        # of qubits actually used in the Circuit.
        # The Circuit qubits attribute is iterated through, with the ith
        # qubit being assigned to the ith qubit of a new "q" register
        passlist.append(FlattenRelabelRegistersPass("q"))
        return SequencePass(passlist)

    def default_compilation_pass(
        self,
        optimisation_level: int = 2,
    ) -> BasePass:
        """
        :param optimisation_level: Allows values of 0,1 or 2, with higher values
            prompting more computationally heavy optimising compilation that
            can lead to reduced gate count in circuits.
        :type optimisation_level: int
        :return: Compilation pass for compiling circuits to Quantinuum devices
        :rtype: BasePass
        """
        return self.default_compilation_pass_with_options(
            optimisation_level, implicit_swap=True, target_2qb_gate=OpType.ZZPhase
        )

    def get_compiled_circuit_with_options(
        self, circuit: Circuit, optimisation_level: int = 2, **kwargs: KwargTypes
    ) -> Circuit:
        """
        Return a single circuit compiled with :py:meth:`default_compilation_pass` See
        :py:meth:`Backend.get_compiled_circuits`.

        Supported kwargs:
        * `implicit_swaps`: Boolean flag, which if true, allows rebasing of
            Circuit via TK2 gates to use implicit wire swaps in circuit
            construction if it reduces the total 2qb qate count.
        * `target_2qb_gate`: pytket OpType, if provided, will rebase
            circuits such that the only two-qubit gates will be of the
            provided type.
        """
        return_circuit = circuit.copy()
        self.default_compilation_pass_with_options(optimisation_level, **kwargs).apply(
            return_circuit
        )
        return return_circuit

    def get_compiled_circuits_with_options(
        self,
        circuits: Sequence[Circuit],
        optimisation_level: int = 2,
        **kwargs: KwargTypes,
    ) -> List[Circuit]:
        """Compile a sequence of circuits with :py:meth:`default_compilation_pass`
        and return the list of compiled circuits (does not act in place).

        As well as applying a degree of optimisation (controlled by the
        `optimisation_level` parameter), this method tries to ensure that the circuits
        can be run on the backend (i.e. successfully passed to
        :py:meth:`process_circuits`), for example by rebasing to the supported gate set,
        or routing to match the connectivity of the device. However, this is not always
        possible, for example if the circuit contains classical operations that are not
        supported by the backend. You may use :py:meth:`valid_circuit` to check whether
        the circuit meets the backend's requirements after compilation. This validity
        check is included in :py:meth:`process_circuits` by default, before any circuits
        are submitted to the backend.

        If the validity check fails, you can obtain more information about the failure
        by iterating through the predicates in the `required_predicates` property of the
        backend, and running the :py:meth:`verify` method on each in turn with your
        circuit.


        :param circuits: The circuits to compile.
        :type circuit: Sequence[Circuit]
        :param optimisation_level: The level of optimisation to perform during
            compilation. See :py:meth:`default_compilation_pass` for a description of
            the different levels (0, 1 or 2). Defaults to 2.
        :type optimisation_level: int, optional
        :return: Compiled circuits.
        :rtype: List[Circuit]


        Supported kwargs:
        * `implicit_swaps`: Boolean flag, which if true, allows rebasing of
            Circuit via TK2 gates to use implicit wire swaps in circuit
            construction if it reduces the total 2qb qate count.
        * `target_2qb_gate`: pytket OpType, if provided, will rebase
            circuits such that the only two-qubit gates will be of the
            provided type.
        """
        return [
            self.get_compiled_circuit_with_options(c, optimisation_level, **kwargs)
            for c in circuits
        ]

    @property
    def _result_id_type(self) -> _ResultIdTuple:
        return tuple((str, str, int))

    @staticmethod
    def _update_result_handle(handle: ResultHandle) -> ResultHandle:
        """Update a legacy handle to be compatible with current format."""
        return handle if len(handle) == 3 else ResultHandle(handle[0], handle[1], -1)

    @staticmethod
    def get_jobid(handle: ResultHandle) -> str:
        """Return the corresponding Quantinuum Job ID from a ResultHandle.

        :param handle: result handle.
        :type handle: ResultHandle
        :return: Quantinuum API Job ID string.
        :rtype: str
        """
        return cast(str, handle[0])

    @staticmethod
    def get_ppcirc_rep(handle: ResultHandle) -> Any:
        """Return the JSON serialization of the classiocal postprocessing circuit
        attached to a handle, if any.

        :param handle: result handle
        :return: serialized post-processing circuit, if any
        """
        return json.loads(cast(str, handle[1]))

    @staticmethod
    def get_results_width(handle: ResultHandle) -> Optional[int]:
        """Return the truncation width of the results, if any.

        :param handle: result handle
        :return: truncation width of results, if any
        """
        n = cast(int, handle[2])
        if n == -1:
            return None
        else:
            assert n >= 0
            return n

    def submit_program(
        self,
        language: Language,
        program: str,
        n_shots: int,
        name: Optional[str] = None,
        noisy_simulation: bool = True,
        group: Optional[str] = None,
        wasm_file_handler: Optional[WasmFileHandler] = None,
        pytket_pass: Optional[BasePass] = None,
        no_opt: bool = False,
        options: Optional[Dict[str, Any]] = None,
        request_options: Optional[Dict[str, Any]] = None,
        results_width: Optional[int] = None,
    ) -> ResultHandle:
        """Submit a program directly to the backend.

        :param program: program (encoded as string)
        :type program: str
        :param language: language
        :type language: Language
        :param n_shots: Number of shots
        :type n_shots: int
        :param name: Job name, defaults to None
        :type name: Optional[str], optional
        :param noisy_simulation: Boolean flag to specify whether the simulator should
          perform noisy simulation with an error model defaults to True
        :type noisy_simulation: bool
        :param group: String identifier of a collection of jobs, can be used for usage
          tracking. Overrides the instance variable `group`, defaults to None
        :type group: Optional[str], optional
        :param wasm_file_handler: ``WasmFileHandler`` object for linked WASM
            module, defaults to None
        :type wasm_file_handler: Optional[WasmFileHandler], optional
        :param no_opt: if true, requests that the backend perform no optimizations
        :type no_opt: bool, defaults to False
        :param pytket_pass: ``pytket.passes.BasePass`` intended to be applied
           by the backend (beta feature, may be ignored), defaults to None
        :type pytket_pass: Optional[BasePass], optional
        :param options: Items to add to the "options" dictionary of the request body
        :type options: Optional[Dict[str, Any]], optional
        :param request_options: Extra options to add to the request body as a
          json-style dictionary, defaults to None
        :type request_options: Optional[Dict[str, Any]], optional
        :param results_width: Number of bits to retain in returned results (if unset,
            retain all)
        :type results_width: Optional[int]
        :raises WasmUnsupported: WASM submitted to backend that does not support it.
        :raises QuantinuumAPIError: API error.
        :raises ConnectionError: Connection to remote API failed
        :return: ResultHandle for submitted job.
        :rtype: ResultHandle
        """

        body: Dict[str, Any] = {
            "name": name or f"{self._label}",
            "count": n_shots,
            "machine": self._device_name,
            "language": language.value,
            "program": program,
            "priority": "normal",
            "options": {
                "simulator": self.simulator_type,
                "no-opt": no_opt,
                "error-model": noisy_simulation,
                "tket": dict(),
            },
        }

        if pytket_pass is not None:
            body["options"]["tket"]["compilation-pass"] = pytket_pass.to_dict()

        group = group or self._group
        if group is not None:
            body["group"] = group

        if wasm_file_handler is not None:
            if self.backend_info and not self.backend_info.misc.get("wasm", False):
                raise WasmUnsupported("Backend does not support wasm calls.")
            body["cfl"] = wasm_file_handler._wasm_file_encoded.decode("utf-8")

        body["options"].update(self._process_circuits_options)
        if options is not None:
            body["options"].update(options)

        # apply any overrides or extra options
        body.update(request_options or {})

        n_bits: int = -1 if results_width is None else results_width

        try:
            res = self.api_handler._submit_job(body)
            if self.api_handler.online:
                jobdict = res.json()
                if res.status_code != HTTPStatus.OK:
                    raise QuantinuumAPIError(
                        f'HTTP error submitting job, {jobdict["error"]}'
                    )
            else:
                return ResultHandle(cast(str, ""), "null", n_bits)
        except ConnectionError:
            raise ConnectionError(
                f"{self._label} Connection Error: Error during submit..."
            )

        # extract job ID from response
        return ResultHandle(cast(str, jobdict["job"]), "null", n_bits)

    def process_circuits(
        self,
        circuits: Sequence[Circuit],
        n_shots: Union[None, int, Sequence[Optional[int]]] = None,
        valid_check: bool = True,
        **kwargs: QuumKwargTypes,
    ) -> List[ResultHandle]:
        """
        See :py:meth:`pytket.backends.Backend.process_circuits`.

        Supported kwargs:

        * `postprocess`: boolean flag to allow classical postprocessing.
        * `noisy_simulation`: boolean flag to specify whether the simulator should
          perform noisy simulation with an error model (default value is `True`).
        * `group`: string identifier of a collection of jobs, can be used for usage
          tracking. Overrides the instance variable `group`.
        * `wasm_file_handler`: a ``WasmFileHandler`` object for linked WASM module.
        * `pytketpass`: a ``pytket.passes.BasePass`` intended to be applied
           by the backend (beta feature, may be ignored).
        * `no_opt`: if true, requests that the backend perform no optimizations
        * `options`: items to add to the "options" dictionary of the request body, as a
          json-style dictionary (in addition to any that were set in the backend
          constructor)
        * `request_options`: extra options to add to the request body as a
          json-style dictionary
        * `language`: languange for submission, of type :py:class:`Language`, default
          QASM.
        * `leakage_detection`: if true, adds additional Qubit and Bit to Circuit
          to detect leakage errors. Run `prune_shots_detected_as_leaky` on returned
          BackendResult to get counts with leakage errors removed.

        """
        circuits = list(circuits)
        n_shots_list = Backend._get_n_shots_as_list(
            n_shots,
            len(circuits),
            optional=False,
        )

        if kwargs.get("leakage_detection", False):
            circuits = [
                self.get_compiled_circuit(
                    get_detection_circuit(c, self.backend_info.n_nodes),  # type: ignore
                    optimisation_level=0,
                )
                for c in circuits
            ]

        if valid_check:
            self._check_all_circuits(circuits)

        postprocess = cast(bool, kwargs.get("postprocess", False))
        noisy_simulation = cast(bool, kwargs.get("noisy_simulation", True))

        group = cast(Optional[str], kwargs.get("group", self._group))

        wasm_fh = cast(Optional[WasmFileHandler], kwargs.get("wasm_file_handler"))

        pytket_pass = cast(Optional[BasePass], kwargs.get("pytketpass"))

        no_opt = cast(bool, kwargs.get("no_opt", False))

        language = cast(Language, kwargs.get("language", Language.QASM))

        handle_list = []

        max_shots = self.backend_info.misc.get("n_shots") if self.backend_info else None
        for circ, n_shots in zip(circuits, n_shots_list):
            if max_shots is not None and n_shots > max_shots:
                raise MaxShotsExceeded(
                    f"Number of shots {n_shots} exceeds maximum {max_shots}"
                )
            if postprocess:
                c0, ppcirc = prepare_circuit(circ, allow_classical=False, xcirc=_xcirc)
                ppcirc_rep = ppcirc.to_dict()
            else:
                c0, ppcirc_rep = circ, None
            n_bits = c0.n_bits
            if language == Language.QASM:
                quantinuum_circ = circuit_to_qasm_str(c0, header="hqslib1")
            else:
                assert language == Language.QIR
                warnings.warn(
                    "Support for Language.QIR is experimental; this may fail!"
                )
                try:
                    pytket_qir_version_components = list(
                        map(int, pytket_qir_version.split(".")[:2])
                    )
                    if (
                        pytket_qir_version_components[0] == 0
                        and pytket_qir_version_components[1] < 2
                    ):
                        raise RuntimeError(
                            "Please install `pytket-qir` version 0.2 or above."
                        )
                    quantinuum_circ = b64encode(
                        cast(
                            bytes,
                            pytket_to_qir(
                                c0,
                                "circuit generated by pytket-qir",
                                QIRFormat.BINARY,
                                True,
                            ),
                        )
                    ).decode("utf-8")
                except NameError:
                    raise RuntimeError(
                        "You must install the `pytket-qir` package in order to use QIR "
                        "submission."
                    )

            if self._MACHINE_DEBUG:
                handle_list.append(
                    ResultHandle(
                        _DEBUG_HANDLE_PREFIX + str((circ.n_qubits, n_shots)),
                        json.dumps(ppcirc_rep),
                        n_bits,
                    )
                )
            else:
                handle = self.submit_program(
                    language,
                    quantinuum_circ,
                    n_shots,
                    name=circ.name or None,
                    noisy_simulation=noisy_simulation,
                    group=group,
                    wasm_file_handler=wasm_fh,
                    pytket_pass=pytket_pass,
                    no_opt=no_opt,
                    options=cast(Dict[str, Any], kwargs.get("options", {})),
                    request_options=cast(
                        Dict[str, Any], kwargs.get("request_options", {})
                    ),
                )

                handle = ResultHandle(
                    self.get_jobid(handle), json.dumps(ppcirc_rep), n_bits
                )
                handle_list.append(handle)
                self._cache[handle] = dict()

        return handle_list

    def _check_batchable(self) -> None:
        if self.backend_info:
            if not self.backend_info.misc.get("batching", False):
                raise BatchingUnsupported()

    def start_batch(
        self,
        max_batch_cost: int,
        circuit: Circuit,
        n_shots: Union[None, int] = None,
        valid_check: bool = True,
        **kwargs: QuumKwargTypes,
    ) -> ResultHandle:
        """Start a batch of jobs on the backend, behaves like `process_circuit`
           but with additional parameter `max_batch_cost` as the first argument.
           See :py:meth:`pytket.backends.Backend.process_circuits` for
           documentation on remaining parameters.


        :param max_batch_cost: Maximum cost to be used for the batch, if a job
            exceeds the batch max it will be rejected.
        :type max_batch_cost: int
        :return: Handle for submitted circuit.
        :rtype: ResultHandle
        """
        self._check_batchable()

        kwargs["request_options"] = {"batch-exec": max_batch_cost}
        [h1] = self.process_circuits([circuit], n_shots, valid_check, **kwargs)

        # make sure the starting job is received, such that subsequent addtions
        # to batch will be recognised as being added to an existing batch
        self.api_handler.retrieve_job_status(
            self.get_jobid(h1),
            use_websocket=cast(bool, kwargs.get("use_websocket", True)),
        )
        return h1

    def add_to_batch(
        self,
        batch_start_job: ResultHandle,
        circuit: Circuit,
        n_shots: Union[None, int] = None,
        batch_end: bool = False,
        valid_check: bool = True,
        **kwargs: QuumKwargTypes,
    ) -> ResultHandle:
        """Add to a batch of jobs on the backend, behaves like `process_circuit`
        except in two ways:\n
        1. The first argument must be the result handle of the first job of
        batch.\n
        2. The optional argument `batch_end` should be set to "True" for the
        final circuit of the batch. By default it is False.\n

        See :py:meth:`pytket.backends.Backend.process_circuits` for
        documentation on remaining parameters.

        :param batch_start_job: Handle of first circuit submitted to batch.
        :type batch_start_job: ResultHandle
        :param batch_end: Boolean flag to signal the final circuit of batch,
            defaults to False
        :type batch_end: bool, optional
        :return: Handle for submitted circuit.
        :rtype: ResultHandle
        """
        self._check_batchable()

        req_opt: Dict[str, Any] = {"batch-exec": self.get_jobid(batch_start_job)}
        if batch_end:
            req_opt["batch-end"] = True
        kwargs["request_options"] = req_opt
        return self.process_circuits([circuit], n_shots, valid_check, **kwargs)[0]

    def _retrieve_job(
        self,
        jobid: str,
        timeout: Optional[int] = None,
        wait: Optional[int] = None,
        use_websocket: Optional[bool] = True,
    ) -> Dict:
        if not self.api_handler:
            raise RuntimeError("API handler not set")
        with self.api_handler.override_timeouts(timeout=timeout, retry_timeout=wait):
            # set and unset optional timeout parameters
            job_dict = self.api_handler.retrieve_job(jobid, use_websocket)

        if job_dict is None:
            raise RuntimeError(f"Unable to retrieve job {jobid}")
        return job_dict

    def cancel(self, handle: ResultHandle) -> None:
        if self.api_handler is not None:
            jobid = self.get_jobid(handle)
            self.api_handler.cancel(jobid)

    def _update_cache_result(self, handle: ResultHandle, res: BackendResult) -> None:
        rescache = {"result": res}

        if handle in self._cache:
            self._cache[handle].update(rescache)
        else:
            self._cache[handle] = rescache

    def circuit_status(
        self, handle: ResultHandle, **kwargs: KwargTypes
    ) -> CircuitStatus:
        handle = self._update_result_handle(handle)
        self._check_handle_type(handle)
        jobid = self.get_jobid(handle)
        if self._MACHINE_DEBUG or jobid.startswith(_DEBUG_HANDLE_PREFIX):
            return CircuitStatus(StatusEnum.COMPLETED)
        use_websocket = cast(bool, kwargs.get("use_websocket", True))
        # TODO check queue position and add to message
        try:
            response = self.api_handler.retrieve_job_status(
                jobid, use_websocket=use_websocket
            )
        except QuantinuumAPIError:
            self.api_handler.login()
            response = self.api_handler.retrieve_job_status(
                jobid, use_websocket=use_websocket
            )

        if response is None:
            raise RuntimeError(f"Unable to retrieve circuit status for handle {handle}")
        circ_status = _parse_status(response)
        if circ_status.status is StatusEnum.COMPLETED:
            if "results" in response:
                ppcirc_rep = self.get_ppcirc_rep(handle)
                n_bits = self.get_results_width(handle)
                ppcirc = (
                    Circuit.from_dict(ppcirc_rep) if ppcirc_rep is not None else None
                )
                self._update_cache_result(
                    handle, _convert_result(response["results"], ppcirc, n_bits)
                )
        return circ_status

    def get_partial_result(
        self, handle: ResultHandle
    ) -> Tuple[Optional[BackendResult], CircuitStatus]:
        """
        Retrieve partial results for a given job, regardless of its current state.

        :param handle: handle to results
        :type handle: ResultHandle

        :return: A tuple containing the results and circuit status.
            If no results are available, the first element is None.
        :rtype: Tuple[Optional[BackendResult], CircuitStatus]
        """
        handle = self._update_result_handle(handle)
        job_id = self.get_jobid(handle)
        jr = self.api_handler.retrieve_job_status(job_id)
        if not jr:
            raise QuantinuumAPIError(f"Unable to retrive job {job_id}")
        res = jr.get("results")
        circ_status = _parse_status(jr)
        if res is None:
            return None, circ_status
        ppcirc_rep = self.get_ppcirc_rep(handle)
        ppcirc = Circuit.from_dict(ppcirc_rep) if ppcirc_rep is not None else None
        n_bits = self.get_results_width(handle)
        backres = _convert_result(res, ppcirc, n_bits)
        return backres, circ_status

    def get_result(self, handle: ResultHandle, **kwargs: KwargTypes) -> BackendResult:
        """
        See :py:meth:`pytket.backends.Backend.get_result`.
        Supported kwargs: `timeout`, `wait`, `use_websocket`.
        """
        handle = self._update_result_handle(handle)
        try:
            return super().get_result(handle)
        except CircuitNotRunError:
            jobid = self.get_jobid(handle)
            ppcirc_rep = self.get_ppcirc_rep(handle)
            n_bits = self.get_results_width(handle)

            ppcirc = Circuit.from_dict(ppcirc_rep) if ppcirc_rep is not None else None

            if self._MACHINE_DEBUG or jobid.startswith(_DEBUG_HANDLE_PREFIX):
                debug_handle_info = jobid[len(_DEBUG_HANDLE_PREFIX) :]
                n_qubits, shots = literal_eval(debug_handle_info)
                return _convert_result(
                    {"c": (["0" * n_qubits] * shots)}, ppcirc, n_bits
                )
            # TODO exception handling when jobid not found on backend
            timeout = kwargs.get("timeout")
            if timeout is not None:
                timeout = int(timeout)
            wait = kwargs.get("wait")
            if wait is not None:
                wait = int(wait)
            use_websocket = cast(Optional[bool], kwargs.get("use_websocket", None))

            job_retrieve = self._retrieve_job(jobid, timeout, wait, use_websocket)
            circ_status = _parse_status(job_retrieve)
            if circ_status.status not in (StatusEnum.COMPLETED, StatusEnum.CANCELLED):
                raise GetResultFailed(
                    f"Cannot retrieve result; job status is {circ_status}"
                )
            try:
                res = job_retrieve["results"]
            except KeyError:
                raise GetResultFailed("Results missing in device return data.")

            backres = _convert_result(res, ppcirc, n_bits)
            self._update_cache_result(handle, backres)
            return backres

    def cost_estimate(self, circuit: Circuit, n_shots: int) -> Optional[float]:
        """Deprecated, use ``cost``."""

        warnings.warn(
            "cost_estimate is deprecated, use cost instead", DeprecationWarning
        )

        return self.cost(circuit, n_shots)

    def cost(
        self,
        circuit: Circuit,
        n_shots: int,
        syntax_checker: Optional[str] = None,
        use_websocket: Optional[bool] = None,
        **kwargs: QuumKwargTypes,
    ) -> Optional[float]:
        """
        Return the cost in HQC to complete this `circuit` with `n_shots`
        repeats.
        If the backend is not a syntax checker (backend name does not end with
        "SC"), it is automatically appended
        to check against the relevant syntax checker.

        See :py:meth:`QuantinuumBackend.process_circuits` for the
        supported kwargs.

        :param circuit: Circuit to calculate runtime estimate for. Must be valid for
            backend.
        :type circuit: Circuit
        :param n_shots: Number of shots.
        :type n_shots: int
        :param syntax_checker: Optional. Name of the syntax checker to use to get cost.
            For example for the "H1-1" device that would be "H1-1SC".
            For most devices this is automatically inferred, default=None.
        :type syntax_checker: str
        :param use_websocket: Optional. Boolean flag to use a websocket connection.
        :type use_websocket: bool
        :raises ValueError: Circuit is not valid, needs to be compiled.
        :return: Cost in HQC to execute the shots.
        :rtype: float
        """
        if not self.valid_circuit(circuit):
            raise ValueError(
                "Circuit does not satisfy predicates of backend."
                + " Try running `backend.get_compiled_circuit` first"
            )

        try:
            syntax_checker = (
                syntax_checker
                or cast(BackendInfo, self.backend_info).misc["syntax_checker"]
            )
        except KeyError:
            raise NoSyntaxChecker(
                "Could not find syntax checker for this backend,"
                " try setting one explicitly with the ``syntax_checker`` parameter"
            )

        backend = QuantinuumBackend(
            cast(str, syntax_checker), api_handler=self.api_handler
        )
        try:
            handle = backend.process_circuit(circuit, n_shots, kwargs=kwargs)  # type: ignore
        except DeviceNotAvailable as e:
            raise ValueError(
                f"Cannot find syntax checker for device {self._device_name}. "
                "Try setting the `syntax_checker` key word argument"
                " to the appropriate syntax checker for"
                " your device explicitly."
            ) from e
        _ = backend.get_result(handle, use_websocket=use_websocket)

        cost = json.loads(
            backend.circuit_status(handle, use_websocket=use_websocket).message
        )["cost"]
        return None if cost is None else float(cost)

    def login(self) -> None:
        """Log in to Quantinuum API. Requests username and password from stdin
        (e.g. shell input or dialogue box in Jupytet notebooks.). Passwords are
        not stored.
        After log in you should not need to provide credentials again while that
        session (script/notebook) is alive.
        """
        self.api_handler.full_login()

    def logout(self) -> None:
        """Clear stored JWT tokens from login. Will need to `login` again to
        make API calls."""
        self.api_handler.delete_authentication()


_xcirc = Circuit(1).add_gate(OpType.PhasedX, [1, 0], [0])
_xcirc.add_phase(0.5)


def _convert_result(
    resultdict: Dict[str, List[str]],
    ppcirc: Optional[Circuit] = None,
    n_bits: Optional[int] = None,
) -> BackendResult:
    array_dict = {
        creg: np.array([list(a) for a in reslist]).astype(np.uint8)
        for creg, reslist in resultdict.items()
    }
    reversed_creg_names = sorted(array_dict.keys(), reverse=True)
    c_bits = [
        Bit(name, ind)
        for name in reversed_creg_names
        for ind in range(array_dict[name].shape[-1] - 1, -1, -1)
    ]
    if n_bits is not None:
        assert n_bits >= 0 and n_bits <= len(c_bits)
        c_bits = c_bits[:n_bits]
        for creg in array_dict.keys():
            array_dict[creg] = array_dict[creg][:, :n_bits]

    stacked_array = np.hstack([array_dict[name] for name in reversed_creg_names])
    return BackendResult(
        c_bits=c_bits,
        shots=OutcomeArray.from_readouts(cast(Sequence[Sequence[int]], stacked_array)),
        ppcirc=ppcirc,
    )


def _parse_status(response: Dict) -> CircuitStatus:
    h_status = response["status"]
    msgdict = {
        k: response.get(k, None)
        for k in (
            "name",
            "submit-date",
            "result-date",
            "queue-position",
            "cost",
            "error",
        )
    }
    message = json.dumps(msgdict)
    return CircuitStatus(_STATUS_MAP[h_status], message)
