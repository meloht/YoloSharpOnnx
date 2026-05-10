using OpenCvSharp;
using System.Security.Cryptography;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Providers;
using YoloSharpOnnx.TestCommon;


namespace YoloSharpOnnx.Test
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
            string imgPath = TestDataUtils.GetImagePathDetect(path);
            string model = TestDataUtils.GetModelPath("yolo11n.onnx");
            using YoloSharp yolo = new YoloSharp(new ExecutionProviderCPU(model));

            var res = yolo.RunDetect(imgPath);
            string ans = res.SummaryOrder();
            Assert.Equal(boxs, ans);

            var res2 = yolo.RunDetectWithTime(imgPath);
            string ans2 = res2.Items.SummaryOrder();
            Assert.Equal(boxs, ans2);
        }

        [Fact]
        public void TestClsYolo11()
        {
            //string imgPath = TestDataUtils.GetImagePath(path);
            //string model = TestDataUtils.GetModelPath("yolo11n.onnx");
            //using YoloSharp yolo = new YoloSharp(new ExecutionProviderCPU(model));

            //var res = yolo.RunDetect(imgPath);
            //string ans = YoloUtils.GetResult(res);
            //Assert.Equal(boxs, ans);

            //var res2 = yolo.RunDetectWithTime(imgPath);
            //string ans2 = YoloUtils.GetResult(res2.Items);
            //Assert.Equal(boxs, ans2);
        }

        [Theory]
        [InlineData(TestDataUtils.Bus, Yolo8.Bus)]
        [InlineData(TestDataUtils.Zidane, Yolo8.Zidane)]
        public void TestDetectYolo8(string path, string boxs)
        {
            string imgPath = TestDataUtils.GetImagePathDetect(path);
            string model = TestDataUtils.GetModelPath("yolov8n.onnx");
            using YoloSharp yolo = new YoloSharp(new ExecutionProviderCPU(model));

            var res = yolo.RunDetect(imgPath);
            string ans = res.SummaryOrder();
            Assert.Equal(boxs, ans);

            var res2 = yolo.RunDetectWithTime(imgPath);
            string ans2 = res2.Items.SummaryOrder();
            Assert.Equal(boxs, ans2);
        }

        [Theory]
        [InlineData(TestDataUtils.Bus, Yolo26.Bus)]
        [InlineData(TestDataUtils.Zidane, Yolo26.Zidane)]
        public void TestDetectYolo26(string path, string boxs)
        {
            string imgPath = TestDataUtils.GetImagePathDetect(path);
            string model = TestDataUtils.GetModelPath("yolo26n.onnx");
            using YoloSharp yolo = new YoloSharp(new ExecutionProviderCPU(model));

            var res = yolo.RunDetect(imgPath);
            string ans = res.SummaryOrder();
            Assert.Equal(boxs, ans);

            var res2 = yolo.RunDetectWithTime(imgPath);
            string ans2 = res2.Items.SummaryOrder();
            Assert.Equal(boxs, ans2);
        }

        [Fact]
        public async Task TestDetectAsyncYolo11()
        {

            string model = TestDataUtils.GetModelPath("yolo11n.onnx");
            using YoloSharp yolo = new YoloSharp(new ExecutionProviderCPU(model));
            using var yoloAsync = yolo.CreateAsyncChannel();

            foreach (var item in _dict)
            {
                var res = await yoloAsync.RunDetectAsync(item.Key);
                Assert.Equal(item.Value, res.SummaryOrder());
            }
            foreach (var item in _dict)
            {
                using var img = Cv2.ImRead(item.Key);
                var res = await yoloAsync.RunDetectAsync(img);
                Assert.Equal(item.Value, res.SummaryOrder());
            }
        }

        [Fact]
        public async Task TestDetectBatchForeachAsync()
        {
            string dir = TestDataUtils.GetImageDirDetect();
            string model = TestDataUtils.GetModelPath("yolo11n.onnx");
            using YoloSharp yolo = new YoloSharp(new ExecutionProviderCPU(model));
            yolo.YoloConfiguration.BatchPoolSize = 4;


            List<string> imgs = TestDataUtils.GetImgPaths();
            int idx = 0;
            await foreach (var item in yolo.BatchDetectForeachAsync(imgs))
            {
                Interlocked.Increment(ref idx);
                Assert.True(_dict.ContainsKey(item.ImagePath));
                Assert.Equal(_dict[item.ImagePath], item.Results.SummaryOrder());
            }

            Assert.Equal(imgs.Count, idx);
        }

        [Fact]
        public void TestDetectBatch()
        {
            string dir = TestDataUtils.GetImageDirDetect();
            string model = TestDataUtils.GetModelPath("yolo11n.onnx");
            using YoloSharp yolo = new YoloSharp(new ExecutionProviderCPU(model));
            yolo.YoloConfiguration.BatchPoolSize = 4;
            var processCallback = new ProcessCallback(_dict);
            var list = yolo.RunBatchDetect(dir, processCallback, ReceiveProcess);


            Assert.Equal(2, list.Length);

            foreach (var item in list)
            {
                Assert.True(_dict.ContainsKey(item.ImagePath));
                Assert.Equal(_dict[item.ImagePath], item.Results.SummaryOrder());
            }

        }


        private void ReceiveProcess(DetectionBatchResult e)
        {
            Assert.True(_dict.ContainsKey(e.ImagePath));
            string res = e.Results.SummaryOrder();
            Assert.Equal(_dict[e.ImagePath], res);
        }

        internal class ProcessCallback : IBatchProcessCallback<DetectionBatchResult>
        {
            private Dictionary<string, string> _dict;
            public ProcessCallback(Dictionary<string, string> dict)
            {
                _dict = dict;
            }
            public void ReceiveProcessResult(DetectionBatchResult e)
            {
                Assert.True(_dict.ContainsKey(e.ImagePath));
                string res = e.Results.SummaryOrder();
                Assert.Equal(_dict[e.ImagePath], res);
            }

        }
    }
}
