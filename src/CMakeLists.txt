add_library(kondo_base OBJECT model.cpp mostovoy_model.cpp dynamics.cpp)

add_executable(test_kpm test_kpm.cpp $<TARGET_OBJECTS:kondo_base>)
target_link_libraries(test_kpm ${LIBS})

add_executable(test_kondo test_kondo.cpp $<TARGET_OBJECTS:kondo_base>)
target_link_libraries(test_kondo ${LIBS})

add_executable(kondo kondo.cpp $<TARGET_OBJECTS:kondo_base>)
target_link_libraries(kondo ${LIBS})

# add_executable(merons merons.cpp $<TARGET_OBJECTS:kondo_base>)
# target_link_libraries(merons ${LIBS})

# add_executable(hongchul hongchul.cpp $<TARGET_OBJECTS:kondo_base>)
# target_link_libraries(hongchul ${LIBS})

add_executable(sc_square sc_square.cpp $<TARGET_OBJECTS:kondo_base>)
target_link_libraries(sc_square ${LIBS})
