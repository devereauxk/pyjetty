#!/bin/bash

function thisdir()
{
        SOURCE="${BASH_SOURCE[0]}"
        while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
          DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
          SOURCE="$(readlink "$SOURCE")"
          [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
        done
        DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
        echo ${DIR}
}
SCRIPTPATH=$(thisdir)
source ${SCRIPTPATH}/../../external/util.sh
separator "pyjetty: ${BASH_SOURCE}"

need_help=$(get_opt "help" $@)
if [ ! -z ${need_help} ]; then
	echo "$0 [--help] [--unsetpyhonpath] [--clean] [--cleanall]"
	exit 0
fi

unsetpyhonpath=$(get_opt "unsetpyhonpath" $@)
if [ ! -z ${unsetpyhonpath} ]; then
    unset PYTHONPATH	&& warning "unsetting PYTHONPATH"
fi

install_path=${SCRIPTPATH}/..
build_path=${SCRIPTPATH}/../build

clean=$(get_opt "clean" $@)
if [ ! -z ${clean} ]; then
    [ -d ${build_path} ] && warning "removing ${build_path}" && rm -rf ${build_path}
	exit 0
fi

cleanall=$(get_opt "cleanall" $@)
if [ ! -z ${cleanall} ]; then
	[ -d ${build_path} ] && warning "removing ${build_path}" && rm -rf ${build_path}
    [ -d ${install_path}/lib ] && warning "removing ${install_path}/lib" && rm -rf ${install_path}/lib
	exit 0
fi

verbose=$(get_opt "verbose" $@)
if [ ! -z ${verbose} ]; then
    export VERBOSE=1
else
    unset VERBOSE
fi


DVENVOPT=""
invenv=$(python -c "import sys; print(hasattr(sys, 'real_prefix'))")
[ "x${invenv}" == "xTrue" ] && DVENVOPT="-DPython3_FIND_VIRTUALENV=ONLY" && warning "- Find/use python in VENV -> option set: ${DVENVOPT}"

build_configuration="Release"
debug=$(get_opt "debug" $@)
if [ ! -z ${debug} ]; then
    build_configuration="Debug"
fi

echo "[i] building in ${build_path}"
mkdir -p ${build_path}
redo=$(get_opt "re" $@)
if [ -d ${build_path} ] || [ "x${redo}" == "xyes" ]; then
	cd ${build_path}
    _wpython=$(which python)
    _python_includes=$(${_wpython} -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")
    _python_libdir=$(${_wpython} -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
    _wpython_config=$(which python-config)
    _python_libs=$(${_wpython_config} --libs)
    _python_numpy_includes=$(${_wpython} -c "import numpy; print(numpy.get_include())")
	cmake -B. -DBUILD_PYTHON=ON -DCMAKE_INSTALL_PREFIX=${install_path} \
    ${DVENVOPT} -DCMAKE_BUILD_TYPE=${build_configuration} \
    -DPYTHON_INCLUDE_DIR=${_python_includes} \
    -DPYTHON_LIBRARY=${_python_libdir} \
    -DPYTHON_LIBS="-L${_python_libdir} ${_python_libs}" \
    -DPython3_NumPy_INCLUDE_DIRS=${_python_numpy_includes} \
    -DPython3_LIBRARIES="-L${_python_libdir} ${_python_libs}" \
    -DPython3_INCLUDE_DIRS=${_python_includes} \
    -DPython_User=TRUE \
    $(abspath ${SCRIPTPATH}/..)
    configure_only=$(get_opt "configure-only" $@)
    if [ ! "x${configure_only}" == "xyes" ]; then
        cmake --build . --target all -- -j $(n_cores) \
	       && cmake --build . --target install
    fi
   cd -
else
	echo "[error] unable to access build path: ${build_path}"
fi
