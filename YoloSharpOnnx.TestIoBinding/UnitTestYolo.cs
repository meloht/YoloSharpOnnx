using YoloSharpOnnx.Providers;
using YoloSharpOnnx.TestCommon;

namespace YoloSharpOnnx.TestIoBinding
{
    public class UnitTestYolo
    {
        private Dictionary<string, string> _dict;
        public UnitTestYolo()
        {
            _dict = TestDataUtils.GetYolo11Dict();
        }

        [Theory]
        [InlineData(TestDataUtils.Bus, Yolo11.Bus)]
        [InlineData(TestDataUtils.Zidane, Yolo11.Zidane)]
        public void TestDetectYolo11(string path, string boxs)
        {
            string imgPath = TestDataUtils.GetImagePath(path);
            string model = TestDataUtils.GetModelPath("yolo11n.onnx");
            using YoloSharp yolo = new YoloSharp(new ExecutionProviderCPU(model));

            var res = yolo.RunDetect(imgPath);
            string ans = YoloUtils.GetResult(res);
            Assert.Equal(boxs, ans);

            var res2 = yolo.RunDetectWithTime(imgPath);
            string ans2 = YoloUtils.GetResult(res2.Items);
            Assert.Equal(boxs, ans2);
        }

        [Theory]
        [InlineData(TestDataUtils.Bus, Yolo8.Bus)]
        [InlineData(TestDataUtils.Zidane, Yolo8.Zidane)]
        public void TestDetectYolo8(string path, string boxs)
        {
            string imgPath = TestDataUtils.GetImagePath(path);
            string model = TestDataUtils.GetModelPath("yolov8n.onnx");
            using YoloSharp yolo = new YoloSharp(new ExecutionProviderCPU(model));

            var res = yolo.RunDetect(imgPath);
            string ans = YoloUtils.GetResult(res);
            Assert.Equal(boxs, ans);

            var res2 = yolo.RunDetectWithTime(imgPath);
            string ans2 = YoloUtils.GetResult(res2.Items);
            Assert.Equal(boxs, ans2);
        }

        [Theory]
        [InlineData(TestDataUtils.Bus, Yolo26.Bus)]
        [InlineData(TestDataUtils.Zidane, Yolo26.Zidane)]
        public void TestDetectYolo26(string path, string boxs)
        {
            string imgPath = TestDataUtils.GetImagePath(path);
            string model = TestDataUtils.GetModelPath("yolo26n.onnx");
            using YoloSharp yolo = new YoloSharp(new ExecutionProviderCPU(model));

            var res = yolo.RunDetect(imgPath);
            string ans = YoloUtils.GetResult(res);
            Assert.Equal(boxs, ans);

            var res2 = yolo.RunDetectWithTime(imgPath);
            string ans2 = YoloUtils.GetResult(res2.Items);
            Assert.Equal(boxs, ans2);
        }

        [Fact]
        public void TestDetectBatch()
        {
            string dir = TestDataUtils.GetImageDir();
            string model = TestDataUtils.GetModelPath("yolo11n.onnx");
            using YoloSharp yolo = new YoloSharp(new ExecutionProviderCPU(model));
            yolo.BatchDetectItemCompleted += Yolo_BatchDetectItemCompleted;

            var list = yolo.RunBatchDetect(dir, 2);


            Assert.Equal(2, list.Length);

            foreach (var item in list)
            {
                Assert.True(_dict.ContainsKey(item.ImagePath));
                Assert.Equal(_dict[item.ImagePath], YoloUtils.GetResult(item.Results));
            }

        }


        private void Yolo_BatchDetectItemCompleted(object? sender, DataResult.DetectionBatchResult e)
        {
            Assert.True(_dict.ContainsKey(e.ImagePath));
            string res = YoloUtils.GetResult(e.Results);
            Assert.Equal(_dict[e.ImagePath], res);
        }


    }
}
