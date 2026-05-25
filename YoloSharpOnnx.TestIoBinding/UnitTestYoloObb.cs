using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Text;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Providers;
using YoloSharpOnnx.TestCommon;

namespace YoloSharpOnnx.TestIoBinding
{
    public class UnitTestYoloObb : IDisposable
    {
        private Dictionary<string, string> _dictObb;

        private YoloSharp yolo11n;
        private YoloSharp yolo8n;
        private YoloSharp yolo26n;
        private int deviceId;
        public UnitTestYoloObb()
        {
            _dictObb = TestDataUtils.GetYolo11ObbDict();
            deviceId = Utils.GetMainGPU();
            yolo11n = new YoloSharp(new ExecutionProviderDirectML(TestDataUtils.GetModelPath("yolo11n-obb.onnx"), deviceId));
            yolo8n = new YoloSharp(new ExecutionProviderDirectML(TestDataUtils.GetModelPath("yolov8n-obb.onnx"), deviceId));
            yolo26n = new YoloSharp(new ExecutionProviderDirectML(TestDataUtils.GetModelPath("yolo26n-obb.onnx"), deviceId));
        }

        [Theory]
        [InlineData(TestDataUtils.Obb01, Yolo11.Obb01)]
        [InlineData(TestDataUtils.Obb02, Yolo11.Obb02)]
        public void TestObbYolo11(string path, string boxs)
        {
            string imgPath = TestDataUtils.GetImagePathObb(path);

            var res = yolo11n.RunObbDetect(imgPath);
            string ans = res.Summary();
            Assert.Equal(boxs, ans);

            var res2 = yolo11n.RunObbDetectWithTime(imgPath);
            string ans2 = res2.Items.Summary();
            Assert.Equal(boxs, ans2);
        }

        [Theory]
        [InlineData(TestDataUtils.Obb01, Yolo8.Obb01)]
        [InlineData(TestDataUtils.Obb02, Yolo8.Obb02)]
        public void TestObbYolo8(string path, string boxs)
        {
            string imgPath = TestDataUtils.GetImagePathObb(path);

            var res = yolo8n.RunObbDetect(imgPath);
            string ans = res.Summary();
            Assert.Equal(boxs, ans);

            var res2 = yolo8n.RunObbDetectWithTime(imgPath);
            string ans2 = res2.Items.Summary();
            Assert.Equal(boxs, ans2);
        }

        [Theory]
        [InlineData(TestDataUtils.Obb01, Yolo26.Obb01)]
        [InlineData(TestDataUtils.Obb02, Yolo26.Obb02)]
        public void TestObbYolo26(string path, string boxs)
        {
            string imgPath = TestDataUtils.GetImagePathObb(path);

            var res = yolo26n.RunObbDetect(imgPath);
            string ans = res.Summary();
            Assert.Equal(boxs, ans);

            var res2 = yolo26n.RunObbDetectWithTime(imgPath);
            string ans2 = res2.Items.Summary();
            Assert.Equal(boxs, ans2);
        }

        [Fact]
        public async Task TestObbAsyncYolo11()
        {

            string model = TestDataUtils.GetModelPath("yolo11n-obb.onnx");
            using YoloSharp yolo = new YoloSharp(new ExecutionProviderDirectML(model, deviceId));
            using var yoloAsync = yolo.CreateAsyncChannel();

            foreach (var item in _dictObb)
            {
                var res = await yoloAsync.RunObbDetectAsync(item.Key);
                Assert.Equal(item.Value, res.Summary());
            }
            foreach (var item in _dictObb)
            {
                using var img = Cv2.ImRead(item.Key);
                var res = await yoloAsync.RunObbDetectAsync(img);
                Assert.Equal(item.Value, res.Summary());
            }
        }

        [Fact]
        public async Task TestObbBatchForeachAsync()
        {
            yolo11n.YoloConfiguration.BatchPoolSize = 4;

            List<string> imgs = TestDataUtils.GetImgObbPaths();
            int idx = 0;
            await foreach (var item in yolo11n.BatchObbDetectForeachAsync(imgs))
            {
                idx++;
                Assert.True(_dictObb.ContainsKey(item.ImagePath));
                Assert.Equal(_dictObb[item.ImagePath], item.Results.Summary());
            }

            Assert.Equal(imgs.Count, idx);
        }

        [Fact]
        public void TestObbBatch()
        {
            string dir = TestDataUtils.GetImageDirObb();

            yolo11n.YoloConfiguration.BatchPoolSize = 4;
            var processCallback = new ProcessCallbackObb(_dictObb);
            var list = yolo11n.RunBatchObbDetect(dir, processCallback, ReceiveProcess);

            Assert.Equal(2, list.Length);

            foreach (var item in list)
            {
                Assert.True(_dictObb.ContainsKey(item.ImagePath));
                Assert.Equal(_dictObb[item.ImagePath], item.Results.Summary());
            }

        }

        private void ReceiveProcess(ObbBatchResult e)
        {
            Assert.True(_dictObb.ContainsKey(e.ImagePath));
            string res = e.Results.Summary();
            Assert.Equal(_dictObb[e.ImagePath], res);
        }

        public void Dispose()
        {
            yolo11n.Dispose();
            yolo26n.Dispose();
            yolo8n.Dispose();
        }

        internal class ProcessCallbackObb : IBatchProcessCallback<ObbBatchResult>
        {
            private Dictionary<string, string> _dict;
            public ProcessCallbackObb(Dictionary<string, string> dict)
            {
                _dict = dict;
            }
            public void ReceiveProcessResult(ObbBatchResult e)
            {
                Assert.True(_dict.ContainsKey(e.ImagePath));
                string res = e.Results.Summary();
                Assert.Equal(_dict[e.ImagePath], res);
            }

        }
    }
}
