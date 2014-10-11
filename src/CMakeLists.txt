add_executable(test_kpm test_kpm.cpp)
target_link_libraries(test_kpm ${LIBS})

add_executable(kondo model.cpp kondo.cpp dynamics.cpp)
target_link_libraries(kondo ${LIBS})

add_executable(test_kondo model.cpp dynamics.cpp test_kondo.cpp)
target_link_libraries(test_kondo ${LIBS})
