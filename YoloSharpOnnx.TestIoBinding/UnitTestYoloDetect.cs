using OpenCvSharp;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Providers;
using YoloSharpOnnx.TestCommon;

namespace YoloSharpOnnx.TestIoBinding
{
    public class UnitTestYoloDetect : IDisposable
    {
        private Dictionary<string, string> _dict;
        private int _deviceId;

        private YoloSharp yolo11n;
        private YoloSharp yolo8n;
        private YoloSharp yolo26n;

        public UnitTestYoloDetect()
        {
            _dict = TestDataUtils.GetYolo11Dict();
            _deviceId = Utils.GetMainGPU();

            yolo11n = new YoloSharp(new ExecutionProviderDirectML(TestDataUtils.GetModelPath("yolo11n.onnx"), _deviceId));
            yolo8n = new YoloSharp(new ExecutionProviderDirectML(TestDataUtils.GetModelPath("yolov8n.onnx"), _deviceId));
            yolo26n = new YoloSharp(new ExecutionProviderDirectML(TestDataUtils.GetModelPath("yolo26n.onnx"), _deviceId));
        }

        [Theory]
        [InlineData(TestDataUtils.Bus, Yolo11.Bus)]
        [InlineData(TestDataUtils.Zidane, Yolo11.Zidane)]
        public void TestDetectYolo11(string path, string boxs)
        {
            string imgPath = TestDataUtils.GetImagePathDetect(path);

            var res = yolo11n.RunDetect(imgPath);
            string ans = res.SummaryOrder();
            Assert.Equal(boxs, ans);

            var res2 = yolo11n.RunDetectWithTime(imgPath);
            string ans2 = res2.Items.SummaryOrder();
            Assert.Equal(boxs, ans2);
        }



        [Theory]
        [InlineData(TestDataUtils.Bus, Yolo8.Bus)]
        [InlineData(TestDataUtils.Zidane, Yolo8.Zidane)]
        public void TestDetectYolo8(string path, string boxs)
        {
            string imgPath = TestDataUtils.GetImagePathDetect(path);

            var res = yolo8n.RunDetect(imgPath);
            string ans = res.SummaryOrder();
            Assert.Equal(boxs, ans);

            var res2 = yolo8n.RunDetectWithTime(imgPath);
            string ans2 = res2.Items.SummaryOrder();
            Assert.Equal(boxs, ans2);
        }


        [Theory]
        [InlineData(TestDataUtils.Bus, Yolo26.Bus)]
        [InlineData(TestDataUtils.Zidane, Yolo26.Zidane)]
        public void TestDetectYolo26(string path, string boxs)
        {
            string imgPath = TestDataUtils.GetImagePathDetect(path);

            var res = yolo26n.RunDetect(imgPath);
            string ans = res.SummaryOrder();
            Assert.Equal(boxs, ans);

            var res2 = yolo26n.RunDetectWithTime(imgPath);
            string ans2 = res2.Items.SummaryOrder();
            Assert.Equal(boxs, ans2);
        }



        [Fact]
        public async Task TestDetectAsyncYolo11()
        {

            string model = TestDataUtils.GetModelPath("yolo11n.onnx");
            using YoloSharp yolo = new YoloSharp(new ExecutionProviderDirectML(model, _deviceId));
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
           
            yolo11n.YoloConfiguration.BatchPoolSize = 4;

            List<string> imgs = TestDataUtils.GetImgPaths();
            int idx = 0;
            await foreach (var item in yolo11n.BatchDetectForeachAsync(imgs))
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

            yolo11n.YoloConfiguration.BatchPoolSize = 4;

            var processCallback = new ProcessCallback(_dict);
            var list = yolo11n.RunBatchDetect(dir, processCallback, ReceiveProcess);

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


        public void Dispose()
        {
            yolo11n.Dispose();
            yolo26n.Dispose();
            yolo8n.Dispose();
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
