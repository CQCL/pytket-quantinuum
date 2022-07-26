Changelog
~~~~~~~~~

0.6.0 (Unreleased)
------------------

* Changed batching interface: `process_circuits` no longer batches, use
  `start_batching` and `add_to_batch` methods to explicitly start and append to
  batches.
* New `submit_qasm` backend method to enable direct submission of a QASM program.

0.5.0 (July 2022)
-----------------

* Updated pytket version requirement to 1.4.
* Add support for multi-factor authentication and microsoft federated login.

0.4.0 (June 2022)
-----------------

* Add wasm support
* Add support for `OpType.CopyBits` and `OpType.ClassicalExpBox` in `QuantinuumBackend`
* Updated pytket version requirement to 1.3.
* Add optional argument `group` to `QuantinuumBackend`

0.3.1 (May 2022)
----------------

* Updated to pyjwt 2.4. This fixes a potential security vulnerability
  (CVE-2022-29217).

0.3.0 (May 2022)
----------------

* ``QuantinuumBackend.cost_estimate`` deprecated, new ``QuantinuumBackend.cost``
  method now uses the syntax checker devices to directly return the cost.
* Updated pytket version requirement to 1.2.

0.2.0 (April 2022)
------------------

* Updated pytket version requirement to 1.1.

0.1.2 (April 2022)
------------------

* Fix batch handling in ``process_circuits()``.

0.1.1 (March 2022)
------------------

* Update device names.


0.1.0 (March 2022)
------------------

* Module renamed from "pytket.extensions.honeywell" to
  "pytket.extensions.quantinumm", with corresponding name changes throughout.
* Simplify authentication: use ``QuantinuumBackend.login()`` to log in once per session.
* Updated pytket version requirement to 1.0.

Old changelog for "pytket-honeywell":

0.21.0 (February 2022)
^^^^^^^^^^^^^^^^^^^^^^

* Updated pytket version requirement to 0.19.
* Drop support for Python 3.7; add support for 3.10.

0.20.0 (January 2022)
^^^^^^^^^^^^^^^^^^^^^

* Added optional ``group`` field to circuit submission.

0.19.0 (January 2022)
^^^^^^^^^^^^^^^^^^^^^

* Updated pytket version requirement to 0.18.

0.18.0 (November 2021)
^^^^^^^^^^^^^^^^^^^^^^

* Updated pytket version requirement to 0.17.

0.17.0 (October 2021)
^^^^^^^^^^^^^^^^^^^^^

* Updated pytket version requirement to 0.16.
* Renamed ``HoneywellBackend.available_devices`` to ``_available_devices`` so as
  not to conflict with abstract ``Backend`` method.

0.16.0 (September 2021)
^^^^^^^^^^^^^^^^^^^^^^^

* Updated pytket version requirement to 0.15.

0.15.0 (September 2021)
^^^^^^^^^^^^^^^^^^^^^^^

* Updated pytket version requirement to 0.14.

0.14.0 (August 2021)
^^^^^^^^^^^^^^^^^^^^

* Support new Honeywell simulator options in :py:class:`HoneywellBackend`:
  "simulator" for simulator type, and "noisy_simulation" to toggle simulations
  with and without error models.
* Device name no longer optional on :py:class:`HoneywellBackend` construction.

0.13.0 (July 2021)
^^^^^^^^^^^^^^^^^^

* Updated pytket version requirement to 0.13.

0.12.0 (June 2021)
^^^^^^^^^^^^^^^^^^

* Updated pytket version requirement to 0.12.

0.11.0 (May 2021)
^^^^^^^^^^^^^^^^^

* Updated pytket version requirement to 0.11.

0.10.0 (May 2021)
^^^^^^^^^^^^^^^^^

* Contextual optimisation added to default compilation passes (except at optimisation level 0).
