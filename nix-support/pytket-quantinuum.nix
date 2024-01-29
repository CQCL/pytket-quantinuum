self: super:
let
  metadata = builtins.readFile ../_metadata.py;
  versions =
    builtins.match ''.*_version__ *= *["']([^"']+)["'].*'' metadata;
  version = if builtins.length versions > 0 then
    builtins.elemAt versions 0
  else
    builtins.trace "Warning: unable to find version. Defaulting to 0.0.0" "0.0.0";
in {
  pytket-quantinuum = super.python3.pkgs.buildPythonPackage {
    pname = "pytket-quantinuum";
    version = version;
    src = super.stdenv.mkDerivation{
      name = "pytket-quantinuum-sources";
      phases = [ "installPhase" ];
      installPhase = ''
        mkdir -p $out;
        cp -r ${../pytket} $out/pytket;
        cp -r ${../tests} $out/tests;

        cp ${../setup.py} $out/setup.py;
        cp ${../README.md} $out/README.md; # required for setup's long description
        cp ${../pytest.ini} $out/pytest.ini;
        cp ${../_metadata.py} $out/_metadata.py;
        
        # on nix versions of scipy and ipython, stubs are missing.
        # adjust mypy.ini to ignore these errors.
        (
          cat ${../mypy.ini};
          cat <<EOF
[mypy-scipy.*]
ignore_missing_imports = True
ignore_errors = True

[mypy-IPython.display.*]
ignore_missing_imports = True
ignore_errors = True
EOF
        ) >> $out/mypy.ini;
      '';
    };
    propagatedBuildInputs = with super.python3Packages; [ msal pyjwt nest-asyncio requests types-requests websockets ] ++ [ super.pytket ];
    checkInputs = with super.python3Packages; [
      mypy
      pytest
      pytest-timeout
      pytest-rerunfailures
      hypothesis
      requests-mock
      llvmlite 
      super.pytket-qir
    ];
    checkPhase = ''
      export HOME=$TMPDIR;

      python -m mypy --config-file=mypy.ini --no-incremental -p pytket -p tests

      cd tests;
      python -m pytest -s .
    '';
  };
}
