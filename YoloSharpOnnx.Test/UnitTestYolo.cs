namespace YoloSharpOnnx.Test
{
    public class UnitTestYolo
    {

        [Theory]
        [InlineData("bus.jpg", "1 bus, 4 person [0.94, 0.9, 0.85, 0.83, 0.4]")]
        [InlineData("zidane.jpg", "2 person, 1 tie [0.86, 0.79, 0.48]")]
        public void TestDetectYolo11(string path, string boxs)
        {
            string imgPath = GetImagePath(path);
            string model = GetModelPath("yolo11n.onnx");
            using YoloSharp _yolo = new YoloSharp(new ExecutionProviderCPU(model));

            var res = _yolo.RunDetect(imgPath);
            string ans = YoloUtils.GetResult(res);
            Assert.Equal(boxs, ans);
        }

        [Theory]
        [InlineData("bus.jpg", "4 person, 1 bus [0.89, 0.88, 0.88, 0.84, 0.44]")]
        [InlineData("zidane.jpg", "2 person [0.83, 0.83]")]
        public void TestDetectYolo8(string path, string boxs)
        {
            string imgPath = GetImagePath(path);
            string model = GetModelPath("yolov8n.onnx");
            using YoloSharp _yolo = new YoloSharp(new ExecutionProviderCPU(model));

            var res = _yolo.RunDetect(imgPath);
            string ans = YoloUtils.GetResult(res);
            Assert.Equal(boxs, ans);
        }

        [Theory]
        [InlineData("bus.jpg", "1 person [0.53]")]
        [InlineData("zidane.jpg", "")]
        public void TestDetectYolo26(string path, string boxs)
        {
            string imgPath = GetImagePath(path);
            string model = GetModelPath("yolo26s.onnx");
            using YoloSharp _yolo = new YoloSharp(new ExecutionProviderCPU(model));

            var res = _yolo.RunDetect(imgPath);
            string ans = YoloUtils.GetResult(res);
            Assert.Equal(boxs, ans);
        }

        private static string GetImagePath(string path)
        {
            return Path.Combine(AppContext.BaseDirectory, "TestData", "Images", path);
        }
        private static string GetModelPath(string path)
        {
            return Path.Combine(AppContext.BaseDirectory, "TestData", "Models", path);
        }
    }
}
