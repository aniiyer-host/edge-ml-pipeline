cc_binary(
    name = "tflm_test",
    srcs = [
        "main.cc",
        "model_data.h",
        "test_data.h",
    ],
    deps = [
        "//tensorflow/lite/micro:micro_framework",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/c:common",
    ],
)