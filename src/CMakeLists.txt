set(KONDO_SRC model.cpp lattice.cpp dynamics.cpp)

add_executable(test_kpm test_kpm.cpp)
target_link_libraries(test_kpm ${LIBS})

add_executable(kondo ${KONDO_SRC} kondo.cpp)
target_link_libraries(kondo ${LIBS})

add_executable(test_kondo ${KONDO_SRC} test_kondo.cpp)
target_link_libraries(test_kondo ${LIBS})
