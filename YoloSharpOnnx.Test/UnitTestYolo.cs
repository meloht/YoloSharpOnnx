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
        private Dictionary<string, string> _dictCls;
        public UnitTestYolo()
        {
            _dict = TestDataUtils.GetYolo11Dict();
            _dictCls= TestDataUtils.GetYolo11ClsDict();
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

        [Theory]
        [InlineData(TestDataUtils.Cls01, Yolo11.Cls01)]
        [InlineData(TestDataUtils.Cls02, Yolo11.Cls02)]
        public void TestClsYolo11(string path, string boxs)
        {
            string imgPath = TestDataUtils.GetImagePathCls(path);
            string model = TestDataUtils.GetModelPath("yolo11n-cls.onnx");
            using YoloSharp yolo = new YoloSharp(new ExecutionProviderCPU(model));

            var res = yolo.RunClassify(imgPath);
            string ans = res.Summary();
            Assert.Equal(boxs, ans);

            var res2 = yolo.RunClassifyWithTime(imgPath);
            string ans2 = res2.Items.Summary();
            Assert.Equal(boxs, ans2);
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
        [InlineData(TestDataUtils.Cls01, Yolo8.Cls01)]
        [InlineData(TestDataUtils.Cls02, Yolo8.Cls02)]
        public void TestClsYolo8(string path, string boxs)
        {
            string imgPath = TestDataUtils.GetImagePathCls(path);
            string model = TestDataUtils.GetModelPath("yolov8n-cls.onnx");
            using YoloSharp yolo = new YoloSharp(new ExecutionProviderCPU(model));

            var res = yolo.RunClassify(imgPath);
            string ans = res.Summary();
            Assert.Equal(boxs, ans);

            var res2 = yolo.RunClassifyWithTime(imgPath);
            string ans2 = res2.Items.Summary();
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


        [Theory]
        [InlineData(TestDataUtils.Cls01, Yolo26.Cls01)]
        [InlineData(TestDataUtils.Cls02, Yolo26.Cls02)]
        public void TestClsYolo26(string path, string boxs)
        {
            string imgPath = TestDataUtils.GetImagePathCls(path);
            string model = TestDataUtils.GetModelPath("yolo26n-cls.onnx");
            using YoloSharp yolo = new YoloSharp(new ExecutionProviderCPU(model));

            var res = yolo.RunClassify(imgPath);
            string ans = res.Summary();
            Assert.Equal(boxs, ans);

            var res2 = yolo.RunClassifyWithTime(imgPath);
            string ans2 = res2.Items.Summary();
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
        public async Task TestClsAsyncYolo11()
        {

            string model = TestDataUtils.GetModelPath("yolo11n-cls.onnx");
            using YoloSharp yolo = new YoloSharp(new ExecutionProviderCPU(model));
            using var yoloAsync = yolo.CreateAsyncChannel();

            foreach (var item in _dictCls)
            {
                var res = await yoloAsync.RunClassifyAsync(item.Key);
                Assert.Equal(item.Value, res.Summary());
            }
            foreach (var item in _dictCls)
            {
                using var img = Cv2.ImRead(item.Key);
                var res = await yoloAsync.RunClassifyAsync(img);
                Assert.Equal(item.Value, res.Summary());
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
                idx++;
                Assert.True(_dict.ContainsKey(item.ImagePath));
                Assert.Equal(_dict[item.ImagePath], item.Results.SummaryOrder());
            }

            Assert.Equal(imgs.Count, idx);
        }

        [Fact]
        public async Task TestClsBatchForeachAsync()
        {
            string dir = TestDataUtils.GetImageDirDetect();
            string model = TestDataUtils.GetModelPath("yolo11n-cls.onnx");
            using YoloSharp yolo = new YoloSharp(new ExecutionProviderCPU(model));
            yolo.YoloConfiguration.BatchPoolSize = 4;


            List<string> imgs = TestDataUtils.GetImgClsPaths();
            int idx = 0;
            await foreach (var item in yolo.BatchClsForeachAsync(imgs))
            {
                idx++;
                Assert.True(_dictCls.ContainsKey(item.ImagePath));
                Assert.Equal(_dictCls[item.ImagePath], item.Results.Summary());
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

        [Fact]
        public void TestClsBatch()
        {
            string dir = TestDataUtils.GetImageDirCls();
            string model = TestDataUtils.GetModelPath("yolo11n-cls.onnx");
            using YoloSharp yolo = new YoloSharp(new ExecutionProviderCPU(model));
            yolo.YoloConfiguration.BatchPoolSize = 4;
            var processCallback = new ProcessCallbackCls(_dictCls);
            var list = yolo.RunBatchCls(dir, processCallback, ReceiveProcess);


            Assert.Equal(2, list.Length);

            foreach (var item in list)
            {
                Assert.True(_dictCls.ContainsKey(item.ImagePath));
                Assert.Equal(_dictCls[item.ImagePath], item.Results.Summary());
            }

        }


        private void ReceiveProcess(DetectionBatchResult e)
        {
            Assert.True(_dict.ContainsKey(e.ImagePath));
            string res = e.Results.SummaryOrder();
            Assert.Equal(_dict[e.ImagePath], res);
        }

        private void ReceiveProcess(ClsBatchResult e)
        {
            Assert.True(_dict.ContainsKey(e.ImagePath));
            string res = e.Results.Summary();
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
        internal class ProcessCallbackCls : IBatchProcessCallback<ClsBatchResult>
        {
            private Dictionary<string, string> _dict;
            public ProcessCallbackCls(Dictionary<string, string> dict)
            {
                _dict = dict;
            }
            public void ReceiveProcessResult(ClsBatchResult e)
            {
                Assert.True(_dict.ContainsKey(e.ImagePath));
                string res = e.Results.Summary();
                Assert.Equal(_dict[e.ImagePath], res);
            }

        }
    }
}
