# Copyright 2020-2022 Cambridge Quantum Computing
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
from dataclasses import dataclass
import json
from http import HTTPStatus
from typing import Dict, List, Set, Optional, Sequence, Union, Any, cast
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
    auto_rebase_pass,
    auto_squash_pass,
)
from pytket.predicates import (  # type: ignore
    GateSetPredicate,
    MaxNQubitsPredicate,
    Predicate,
    NoSymbolsPredicate,
)
from pytket.utils import prepare_circuit
from pytket.utils.outcomearray import OutcomeArray
from pytket.wasm import WasmFileHandler

from pytket.extensions.quantinuum.backends.credential_storage import (
    MemoryCredentialStorage,
)

from .api_wrappers import QuantinuumAPIError, QuantinuumAPI

_DEBUG_HANDLE_PREFIX = "_MACHINE_DEBUG_"
QUANTINUUM_URL_PREFIX = "https://qapi.quantinuum.com/"
DEVICE_FAMILY = "H1"
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


# DEFAULT_CREDENTIALS_STORAGE for use with the DEFAULT_API_HANDLER.
DEFAULT_CREDENTIALS_STORAGE = MemoryCredentialStorage()

# DEFAULT_API_HANDLER provides a global re-usable API handler
# that will persist after this module is imported.
#
# This allows users to create multiple QuantinuumBackend instances
# without requiring them to acquire new tokens.
DEFAULT_API_HANDLER = QuantinuumAPI(DEFAULT_CREDENTIALS_STORAGE)

QuumKwargTypes = Union[KwargTypes, WasmFileHandler, Dict[str, Any]]


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

        :param device_name: Name of device, e.g. "H1"
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
        gate_set: List[str] = dct.pop("gateset", [])
        return BackendInfo(
            name=cls.__name__,
            device_name=name,
            version=__extension_version__,
            architecture=FullyConnected(n_qubits),
            gate_set=_get_gateset(gate_set),
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
        jr = res.json()
        try:
            return str(jr["state"])
        except KeyError:
            # for family backends the response dictionary is different
            # {"<device_name>": <state>}
            return str(jr)

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

        return preds

    def rebase_pass(self) -> BasePass:
        return auto_rebase_pass(self._gate_set)

    def default_compilation_pass(self, optimisation_level: int = 1) -> BasePass:
        assert optimisation_level in range(3)
        passlist = [
            DecomposeBoxes(),
            scratch_reg_resize_pass(),
        ]
        squash = auto_squash_pass({OpType.PhasedX, OpType.Rz})

        # use default (perfect fidelities) for supported gates
        fidelities: Dict[str, Any] = {}
        if OpType.ZZMax in self._gate_set:
            fidelities["ZZMax_fidelity"] = 1.0
        if OpType.ZZPhase in self._gate_set:
            fidelities["ZZPhase_fidelity"] = lambda x: 1.0
        if len(fidelities) == 0:
            raise QuantinuumAPIError(
                "Either ZZMax or ZZPhase gate must be supported by device"
            )
        # If you make changes to the default_compilation_pass,
        # then please update this page accordingly
        # https://cqcl.github.io/pytket-quantinuum/api/index.html#default-compilation
        # Edit this docs source file -> pytket-quantinuum/docs/intro.txt
        if optimisation_level == 0:
            return SequencePass(passlist + [self.rebase_pass()])
        elif optimisation_level == 1:
            return SequencePass(
                passlist
                + [
                    SynthesiseTK(),
                    NormaliseTK2(),
                    DecomposeTK2(**fidelities),
                    self.rebase_pass(),
                    ZZPhaseToRz(),
                    RemoveRedundancies(),
                    squash,
                    SimplifyInitial(
                        allow_classical=False, create_all_qubits=True, xcirc=_xcirc
                    ),
                ]
            )
        else:
            return SequencePass(
                passlist
                + [
                    FullPeepholeOptimise(target_2qb_gate=OpType.TK2),
                    NormaliseTK2(),
                    DecomposeTK2(**fidelities),
                    self.rebase_pass(),
                    RemoveRedundancies(),
                    squash,
                    SimplifyInitial(
                        allow_classical=False, create_all_qubits=True, xcirc=_xcirc
                    ),
                ]
            )

    @property
    def _result_id_type(self) -> _ResultIdTuple:
        return tuple((str, str))

    def get_jobid(self, handle: ResultHandle) -> str:
        """Return the corresponding Quantinuum Job ID from a ResultHandle.

        :param handle: result handle.
        :type handle: ResultHandle
        :return: Quantinuum API Job ID string.
        :rtype: str
        """
        return cast(str, handle[0])

    def submit_qasm(
        self,
        qasm: str,
        n_shots: int,
        name: Optional[str] = None,
        noisy_simulation: bool = True,
        group: Optional[str] = None,
        wasm_file_handler: Optional[WasmFileHandler] = None,
        pytket_pass: Optional[BasePass] = None,
        options: Optional[Dict[str, Any]] = None,
        request_options: Optional[Dict[str, Any]] = None,
    ) -> ResultHandle:
        """Submit a qasm program directly to the backend.

        :param qasm: QASM 2.0 program.
        :type qasm: str
        :param n_shots: Number of shots
        :type n_shots: int
        :param name: Job name, defaults to None
        :type name: Optional[str], optional
        :param noisy_simulation: Boolean flag to specify whether the simulator should
          perform noisy simulation with an error model defaults to True
        :type noisy_simulation: bool, optional
        :param group: String identifier of a collection of jobs, can be used for usage
          tracking. Overrides the instance variable `group`, defaults to None
        :type group: Optional[str], optional
        :param wasm_file_handler: ``WasmFileHandler`` object for linked WASM
            module, defaults to None
        :type wasm_file_handler: Optional[WasmFileHandler], optional
        :param pytket_pass: ``pytket.passes.BasePass`` intended to be applied
           by the backend (beta feature, may be ignored), defaults to None
        :type pytket_pass: Optional[BasePass], optional
        :param options: Items to add to the "options" dictionary of the request body
        :type options: Optional[Dict[str, Any]], optional
        :param request_options: Extra options to add to the request body as a
          json-style dictionary, defaults to None
        :type request_options: Optional[Dict[str, Any]], optional
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
            "language": "OPENQASM 2.0",
            "program": qasm,
            "priority": "normal",
            "options": {
                "simulator": self.simulator_type,
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

        try:
            res = self.api_handler._submit_job(body)
            if self.api_handler.online:
                jobdict = res.json()
                if res.status_code != HTTPStatus.OK:
                    raise QuantinuumAPIError(
                        f'HTTP error submitting job, {jobdict["error"]}'
                    )
            else:
                return ResultHandle(cast(str, ""), "null")
        except ConnectionError:
            raise ConnectionError(
                f"{self._label} Connection Error: Error during submit..."
            )

        # extract job ID from response
        return ResultHandle(cast(str, jobdict["job"]), "null")

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
        * `options`: items to add to the "options" dictionary of the request body, as a
          json-style dictionary (in addition to any that were set in the backend
          constructor)
        * `request_options`: extra options to add to the request body as a
          json-style dictionary

        """
        circuits = list(circuits)
        n_shots_list = Backend._get_n_shots_as_list(
            n_shots,
            len(circuits),
            optional=False,
        )

        if valid_check:
            self._check_all_circuits(circuits)

        postprocess = cast(bool, kwargs.get("postprocess", False))
        noisy_simulation = cast(bool, kwargs.get("noisy_simulation", True))

        group = cast(Optional[str], kwargs.get("group", self._group))

        wasm_fh = cast(Optional[WasmFileHandler], kwargs.get("wasm_file_handler"))

        pytket_pass = cast(Optional[BasePass], kwargs.get("pytketpass"))

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
            quantinuum_circ = circuit_to_qasm_str(c0, header="hqslib1")

            if self._MACHINE_DEBUG:
                handle_list.append(
                    ResultHandle(
                        _DEBUG_HANDLE_PREFIX + str((circ.n_qubits, n_shots)),
                        json.dumps(ppcirc_rep),
                    )
                )
            else:
                handle = self.submit_qasm(
                    quantinuum_circ,
                    n_shots,
                    name=circ.name or None,
                    noisy_simulation=noisy_simulation,
                    group=group,
                    wasm_file_handler=wasm_fh,
                    pytket_pass=pytket_pass,
                    options=cast(Dict[str, Any], kwargs.get("options", {})),
                    request_options=cast(
                        Dict[str, Any], kwargs.get("request_options", {})
                    ),
                )

                handle = ResultHandle(handle[0], json.dumps(ppcirc_rep))
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
        [
            h1,
        ] = self.process_circuits([circuit], n_shots, valid_check, **kwargs)

        # make sure the starting job is received, such that subsequent addtions
        # to batch will be recognised as being added to an existing batch
        self.api_handler.retrieve_job_status(
            str(h1[0]),
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
            jobid = str(handle[0])
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
        self._check_handle_type(handle)
        jobid = str(handle[0])
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
                ppcirc_rep = json.loads(cast(str, handle[1]))
                ppcirc = (
                    Circuit.from_dict(ppcirc_rep) if ppcirc_rep is not None else None
                )
                self._update_cache_result(
                    handle, _convert_result(response["results"], ppcirc)
                )
        return circ_status

    def get_result(self, handle: ResultHandle, **kwargs: KwargTypes) -> BackendResult:
        """
        See :py:meth:`pytket.backends.Backend.get_result`.
        Supported kwargs: `timeout`, `wait`, `use_websocket`.
        """
        try:
            return super().get_result(handle)
        except CircuitNotRunError:
            jobid = str(handle[0])
            ppcirc_rep = json.loads(cast(str, handle[1]))
            ppcirc = Circuit.from_dict(ppcirc_rep) if ppcirc_rep is not None else None

            if self._MACHINE_DEBUG or jobid.startswith(_DEBUG_HANDLE_PREFIX):
                debug_handle_info = jobid[len(_DEBUG_HANDLE_PREFIX) :]
                n_qubits, shots = literal_eval(debug_handle_info)
                return _convert_result({"c": (["0" * n_qubits] * shots)}, ppcirc)
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

            backres = _convert_result(res, ppcirc)
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
    ) -> Optional[float]:
        """
        Return the cost in HQC to complete this `circuit` with `n_shots`
        repeats.
        If the backend is not a syntax checker (backend name does not end with
        "SC"), it is automatically appended
        to check against the relevant syntax checker.
        Sometimes it may not be possible to find the relevant syntax checker,
        for example for device families. In which case you may need to set
        the ``syntax_checker`` kwarg to the appropriate syntax checker name.

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
            handle = backend.process_circuit(circuit, n_shots)
        except DeviceNotAvailable as e:
            raise ValueError(
                f"Cannot find syntax checker for device {self._device_name}. "
                "Try setting the `syntax_checker` key word argument"
                " to the appropriate syntax checker for"
                " your device explicitly. "
                "For device families, you may need to pick the"
                " syntax checker for the specific device,"
                " e.g. 'H1-1SC' as opposed to 'H1SC'"
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
    resultdict: Dict[str, List[str]], ppcirc: Optional[Circuit] = None
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
