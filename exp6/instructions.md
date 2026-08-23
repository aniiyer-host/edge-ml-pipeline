### place tflm_test into tflite-micro and run:
```bash
bazelisk build //tflm_test:tflm_test \
  --distdir="$PWD/../bazel-downloads"

./bazel-bin/tflm_test/tflm_test
```

### Note: bazel encountered some network proxy issues hence had to download dependencies manually as the build failed.