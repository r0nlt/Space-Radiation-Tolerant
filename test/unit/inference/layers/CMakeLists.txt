add_executable(conv_layer_test conv_layer_test.cpp)
target_link_libraries(conv_layer_test PRIVATE rad_ml)

add_test(NAME ConvLayerTest COMMAND conv_layer_test) 