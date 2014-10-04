add_executable(test_kpm test_kpm.cpp)
target_link_libraries(test_kpm ${LIBS})

add_executable(kondo hamiltonian.cpp kondo.cpp)
target_link_libraries(kondo ${LIBS})
