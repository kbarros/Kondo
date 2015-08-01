add_executable(test_kpm test_kpm.cpp)
target_link_libraries(test_kpm ${LIBS})

add_executable(test_BdG test_BdG.cpp)
target_link_libraries(test_BdG ${LIBS})


add_library(kondo_base OBJECT model.cpp mostovoy_model.cpp dynamics.cpp)

add_executable(test_kondo test_kondo.cpp $<TARGET_OBJECTS:kondo_base>)
target_link_libraries(test_kondo ${LIBS})

add_executable(kondo kondo.cpp $<TARGET_OBJECTS:kondo_base>)
target_link_libraries(kondo ${LIBS})
