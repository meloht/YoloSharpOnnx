using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Text;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Providers;
using YoloSharpOnnx.TestCommon;

namespace YoloSharpOnnx.Test
{

    public class UnitTestYoloClassify : IDisposable
    {
        private Dictionary<string, string> _dictCls;

        private YoloSharp yolo11n;
        private YoloSharp yolo8n;
        private YoloSharp yolo26n;

        public UnitTestYoloClassify()
        {
            _dictCls = TestDataUtils.GetYolo11ClsDict();
            yolo11n = new YoloSharp(new ExecutionProviderCPU(TestDataUtils.GetModelPath("yolo11n-cls.onnx")));
            yolo8n = new YoloSharp(new ExecutionProviderCPU(TestDataUtils.GetModelPath("yolov8n-cls.onnx")));
            yolo26n = new YoloSharp(new ExecutionProviderCPU(TestDataUtils.GetModelPath("yolo26n-cls.onnx")));
        }

        [Theory]
        [InlineData(TestDataUtils.Cls01, Yolo11.Cls01)]
        [InlineData(TestDataUtils.Cls02, Yolo11.Cls02)]
        public void TestClsYolo11(string path, string boxs)
        {
            string imgPath = TestDataUtils.GetImagePathCls(path);

            var res = yolo11n.RunClassify(imgPath);
            string ans = res.Summary();
            Assert.Equal(boxs, ans);

            var res2 = yolo11n.RunClassifyWithTime(imgPath);
            string ans2 = res2.Items.Summary();
            Assert.Equal(boxs, ans2);
        }

        [Theory]
        [InlineData(TestDataUtils.Cls01, Yolo8.Cls01)]
        [InlineData(TestDataUtils.Cls02, Yolo8.Cls02)]
        public void TestClsYolo8(string path, string boxs)
        {
            string imgPath = TestDataUtils.GetImagePathCls(path);
           
            var res = yolo8n.RunClassify(imgPath);
            string ans = res.Summary();
            Assert.Equal(boxs, ans);

            var res2 = yolo8n.RunClassifyWithTime(imgPath);
            string ans2 = res2.Items.Summary();
            Assert.Equal(boxs, ans2);
        }

        [Theory]
        [InlineData(TestDataUtils.Cls01, Yolo26.Cls01)]
        [InlineData(TestDataUtils.Cls02, Yolo26.Cls02)]
        public void TestClsYolo26(string path, string boxs)
        {
            string imgPath = TestDataUtils.GetImagePathCls(path);
           
            var res = yolo26n.RunClassify(imgPath);
            string ans = res.Summary();
            Assert.Equal(boxs, ans);

            var res2 = yolo26n.RunClassifyWithTime(imgPath);
            string ans2 = res2.Items.Summary();
            Assert.Equal(boxs, ans2);
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
        public async Task TestClsBatchForeachAsync()
        {
            yolo11n.YoloConfiguration.BatchPoolSize = 4;

            List<string> imgs = TestDataUtils.GetImgClsPaths();
            int idx = 0;
            await foreach (var item in yolo11n.BatchClsForeachAsync(imgs))
            {
                idx++;
                Assert.True(_dictCls.ContainsKey(item.ImagePath));
                Assert.Equal(_dictCls[item.ImagePath], item.Results.Summary());
            }

            Assert.Equal(imgs.Count, idx);
        }

        [Fact]
        public void TestClsBatch()
        {
            string dir = TestDataUtils.GetImageDirCls();

            yolo11n.YoloConfiguration.BatchPoolSize = 4;
            var processCallback = new ProcessCallbackCls(_dictCls);
            var list = yolo11n.RunBatchCls(dir, processCallback, ReceiveProcess);

            Assert.Equal(2, list.Length);

            foreach (var item in list)
            {
                Assert.True(_dictCls.ContainsKey(item.ImagePath));
                Assert.Equal(_dictCls[item.ImagePath], item.Results.Summary());
            }

        }

        private void ReceiveProcess(ClsBatchResult e)
        {
            Assert.True(_dictCls.ContainsKey(e.ImagePath));
            string res = e.Results.Summary();
            Assert.Equal(_dictCls[e.ImagePath], res);
        }

        public void Dispose()
        {
            yolo11n.Dispose();
            yolo26n.Dispose();
            yolo8n.Dispose();
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
