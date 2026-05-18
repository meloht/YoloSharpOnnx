using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Text;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Providers;
using YoloSharpOnnx.TestCommon;

namespace YoloSharpOnnx.TestIoBinding
{
    public class UnitTestYoloSegment : IDisposable
    {
        private Dictionary<string, string> _dictSeg;

        private YoloSharp yolo11n;
        private YoloSharp yolo8n;
        private YoloSharp yolo26n;
        private int deviceId;

        public UnitTestYoloSegment()
        {
            _dictSeg = TestDataUtils.GetYolo11SegDict();
            deviceId = Utils.GetMainGPU();

            yolo11n = new YoloSharp(new ExecutionProviderDirectML(TestDataUtils.GetModelPath("yolo11n-seg.onnx"), deviceId));
            yolo8n = new YoloSharp(new ExecutionProviderDirectML(TestDataUtils.GetModelPath("yolov8n-seg.onnx"), deviceId));
            yolo26n = new YoloSharp(new ExecutionProviderDirectML(TestDataUtils.GetModelPath("yolo26n-seg.onnx"), deviceId));
        }

        [Theory]
        [InlineData(TestDataUtils.Seg01, Yolo11.Seg01)]
        [InlineData(TestDataUtils.Seg02, Yolo11.Seg02)]
        public void TestSegYolo11(string path, string boxs)
        {
            string imgPath = TestDataUtils.GetImagePathSeg(path);

            var res = yolo11n.RunSegment(imgPath);
            string ans = res.Summary();
            Assert.Equal(boxs, ans);

            var res2 = yolo11n.RunSegmentWithTime(imgPath);
            string ans2 = res2.Items.Summary();
            Assert.Equal(boxs, ans2);
        }

        [Theory]
        [InlineData(TestDataUtils.Seg01, Yolo8.Seg01)]
        [InlineData(TestDataUtils.Seg02, Yolo8.Seg02)]
        public void TestSegYolo8(string path, string boxs)
        {
            string imgPath = TestDataUtils.GetImagePathSeg(path);

            var res = yolo8n.RunSegment(imgPath);
            string ans = res.Summary();
            Assert.Equal(boxs, ans);

            var res2 = yolo8n.RunSegmentWithTime(imgPath);
            string ans2 = res2.Items.Summary();
            Assert.Equal(boxs, ans2);
        }

        [Theory]
        [InlineData(TestDataUtils.Seg01, Yolo26.Seg01)]
        [InlineData(TestDataUtils.Seg02, Yolo26.Seg02)]
        public void TestSegYolo26(string path, string boxs)
        {
            string imgPath = TestDataUtils.GetImagePathSeg(path);

            var res = yolo26n.RunSegment(imgPath);
            string ans = res.Summary();
            Assert.Equal(boxs, ans);

            var res2 = yolo26n.RunSegmentWithTime(imgPath);
            string ans2 = res2.Items.Summary();
            Assert.Equal(boxs, ans2);
        }

        [Fact]
        public async Task TestSegAsyncYolo11()
        {

            string model = TestDataUtils.GetModelPath("yolo11n-seg.onnx");
            using YoloSharp yolo = new YoloSharp(new ExecutionProviderDirectML(model, deviceId));
            using var yoloAsync = yolo.CreateAsyncChannel();

            foreach (var item in _dictSeg)
            {
                var res = await yoloAsync.RunSegmentAsync(item.Key);
                Assert.Equal(item.Value, res.Summary());
            }
            foreach (var item in _dictSeg)
            {
                using var img = Cv2.ImRead(item.Key);
                var res = await yoloAsync.RunSegmentAsync(img);
                Assert.Equal(item.Value, res.Summary());
            }
        }

        [Fact]
        public async Task TestSegBatchForeachAsync()
        {
            yolo11n.YoloConfiguration.BatchPoolSize = 4;

            List<string> imgs = TestDataUtils.GetImgSegPaths();
            int idx = 0;
            await foreach (var item in yolo11n.BatchSegmentForeachAsync(imgs))
            {
                idx++;
                Assert.True(_dictSeg.ContainsKey(item.ImagePath));
                Assert.Equal(_dictSeg[item.ImagePath], item.Results.Summary());
            }

            Assert.Equal(imgs.Count, idx);
        }

        [Fact]
        public void TestSegBatch()
        {
            string dir = TestDataUtils.GetImageDirSeg();

            yolo11n.YoloConfiguration.BatchPoolSize = 4;
            var processCallback = new ProcessCallbackSeg(_dictSeg);
            var list = yolo11n.RunBatchSegment(dir, processCallback, ReceiveProcess);

            Assert.Equal(2, list.Length);

            foreach (var item in list)
            {
                Assert.True(_dictSeg.ContainsKey(item.ImagePath));
                Assert.Equal(_dictSeg[item.ImagePath], item.Results.Summary());
            }

        }

        private void ReceiveProcess(SegBatchResult e)
        {
            Assert.True(_dictSeg.ContainsKey(e.ImagePath));
            string res = e.Results.Summary();
            Assert.Equal(_dictSeg[e.ImagePath], res);
        }

        public void Dispose()
        {
            yolo11n.Dispose();
            yolo26n.Dispose();
            yolo8n.Dispose();
        }

        internal class ProcessCallbackSeg : IBatchProcessCallback<SegBatchResult>
        {
            private Dictionary<string, string> _dict;
            public ProcessCallbackSeg(Dictionary<string, string> dict)
            {
                _dict = dict;
            }
            public void ReceiveProcessResult(SegBatchResult e)
            {
                Assert.True(_dict.ContainsKey(e.ImagePath));
                string res = e.Results.Summary();
                Assert.Equal(_dict[e.ImagePath], res);
            }

        }
    }
}
