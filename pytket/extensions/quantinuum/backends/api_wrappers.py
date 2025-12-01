# Copyright Quantinuum
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

from typing import Any


class QuantinuumAPIError(Exception):
    pass


OFFLINE_MACHINE_LIST = [
    {
        "wasm": True,
        "batching": True,
        "supported_languages": ["OPENQASM 2.0", "QIR 1.0"],
        "benchmarks": {"qv": {"date": "2024-04-04", "value": 1048576.0}},
        "max_classical_register_width": 63,
        "gateset": ["RZZ", "Rxxyyzz", "Rz", "U1q", "ZZ"],
        "name": "H1-1",
        "syntax_checker": "H1-1SC",
        "n_gate_zones": "5",
        "noise_specs": {
            "date": "2025-05-02",
            "spam_error": {
                "p_meas_1_unc": 0.000176,
                "p_meas_0": 0.00122,
                "p_meas_1": 0.00343,
                "p_meas_0_unc": 0.000105,
            },
            "crosstalk_error": {
                "p_crosstalk_meas_unc": 1.7e-06,
                "p_crosstalk_meas": 4.1e-05,
            },
            "memory_error": {
                "memory_error_unc": 9.16e-05,
                "memory_error": 0.000222,
            },
            "1q_gate_error": {"p1": 1.8e-05, "p1_unc": 2.93e-06},
            "2q_gate_error": {"p2_unc": 6.23e-05, "p2": 0.000973},
        },
        "max_n_shots": 10000,
        "n_qubits": 20,
        "n_classical_registers": 4000,
        "system_type": "hardware",
        "connectivity": "all-to-all",
        "emulator": "H1-1E",
    },
    {
        "wasm": True,
        "batching": True,
        "supported_languages": ["OPENQASM 2.0", "QIR 1.0"],
        "benchmarks": {"qv": {"date": "2024-08-11", "value": 2097152.0}},
        "max_classical_register_width": 63,
        "gateset": ["RZZ", "Rxxyyzz", "Rz", "U1q", "ZZ"],
        "name": "H2-1",
        "syntax_checker": "H2-1SC",
        "n_gate_zones": "4",
        "noise_specs": {
            "date": "2025-04-30",
            "spam_error": {
                "p_meas_1_unc": 0.000124,
                "p_meas_0": 0.0006,
                "p_meas_1": 0.00139,
                "p_meas_0_unc": 8.16e-05,
            },
            "crosstalk_error": {
                "p_crosstalk_meas_unc": 8.98e-07,
                "p_crosstalk_meas": 6.65e-06,
            },
            "memory_error": {
                "memory_error_unc": 2.34e-05,
                "memory_error": 0.000203,
            },
            "1q_gate_error": {"p1": 1.89e-05, "p1_unc": 4.23e-06},
            "2q_gate_error": {"p2_unc": 8.08e-05, "p2": 0.00105},
        },
        "max_n_shots": 10000,
        "n_qubits": 56,
        "n_classical_registers": 4000,
        "system_type": "hardware",
        "connectivity": "all-to-all",
        "emulator": "H2-1E",
    },
    {
        "wasm": True,
        "batching": True,
        "supported_languages": ["OPENQASM 2.0", "QIR 1.0"],
        "benchmarks": {"qv": {"date": "2025-09-08", "value": 33554432.0}},
        "max_classical_register_width": 63,
        "gateset": ["RZZ", "Rxxyyzz", "Rz", "U1q", "ZZ"],
        "name": "H2-2",
        "syntax_checker": "H2-2SC",
        "n_gate_zones": "4",
        "noise_specs": {
            "date": "2025-08-28",
            "spam_error": {
                "p_meas_1_unc": 0.00011,
                "p_meas_0": 0.00067,
                "p_meas_1": 0.0012,
                "p_meas_0_unc": 8.7e-05,
            },
            "crosstalk_error": {
                "p_crosstalk_meas_unc": 5.3e-07,
                "p_crosstalk_meas": 2.2e-05,
            },
            "memory_error": {"memory_error_unc": 2e-05, "memory_error": 0.00012},
            "1q_gate_error": {"p1": 2.8e-05, "p1_unc": 3.6e-06},
            "2q_gate_error": {"p2_unc": 4.8e-05, "p2": 0.00083},
        },
        "max_n_shots": 10000,
        "n_qubits": 56,
        "n_classical_registers": 4000,
        "system_type": "hardware",
        "connectivity": "all-to-all",
        "emulator": "H2-2E",
    },
]


class QuantinuumAPIOffline:
    """
    Offline Quantinuum API emulator.
    """

    def __init__(self, machine_list: list | None = None):
        """Initialize offline API client.

        All jobs that are submitted to this offline API are stored
        and can be requested again later.

        :param machine_list: List of dictionaries each containing device information.
            One short example:
            {
            "name": "H1-1",
            "n_qubits": 20,
            "gateset": ["RZZ", "Riswap", "TK2"],
            "n_shots": 10000,
            "batching": True,
            }
        """
        if machine_list is None:
            machine_list = OFFLINE_MACHINE_LIST
        self.machine_list = machine_list
        self.submitted: list = []

    def get_machine_list(self) -> list[dict[str, Any]]:
        """Returns a given list of the available machines
        :return: list of machines
        """

        return self.machine_list

    def _submit_job(self, body: dict) -> None:
        """The function will take the submitted job and store it for later

        :param body: submitted job

        :return: None
        """
        self.submitted.append(body)

    def get_jobs(self) -> list | None:
        """The function will return all the jobs that have been submitted

        :return: List of all the submitted jobs
        """
        return self.submitted


class QuantinuumAPI(QuantinuumAPIOffline):
    """
    Alias for `QuantinuumAPIOffline`.
    """
