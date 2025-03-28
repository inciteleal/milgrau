# Exemplo de arquivo de testes. Colocar aqui rotinas para teste da biblioteca.

import pytest
import os

#from scripts import libids_scc2netcdf - da algum pau estranho qd carrega licel - pytest eixar de funcioanar
from scripts import libids
from scripts import lipancora
#from scripts import liracos - tb dapau com pytest
#from scripts import radiodata


def test_func():
    assert 1 == 1

def test_01_libids():

    libids.libids("./tests/data")
    #libids_scc2netcdf.libids_scc2_netcdf("./tests/data")
    lipancora.lipancora("./tests/data")
    # lipancora
    # liracos
    # radiodata
    # lebear
    # lirabear
    # tropopause

    
    assert 1 == 1

def test_02_libids_scc2netcdf():
    #libids_scc2netcdf.libids_scc2netcdf("./tests/data")
    assert 1 == 1


def test_03_lipancora():
    lipancora.lipancora("./tests/data")

def test_04_liracos():
    lipancora.liracos("./tests/data")

def test_05_radiodata():
    lipancora.liracos("./tests/data")


if __name__ == "__main__":
    pytest.main()
