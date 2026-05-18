using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Text;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Providers;
using YoloSharpOnnx.TestCommon;

namespace YoloSharpOnnx.TestIoBinding
{
    public class UnitTestSegmentAPI : IDisposable
    {
        private Dictionary<string, string> _dictSeg;
        private string model;
        private YoloSharp yolo;

        public UnitTestSegmentAPI()
        {
            _dictSeg = TestDataUtils.GetYolo11SegDict();
            model = TestDataUtils.GetModelPath("yolo11n-seg.onnx");
            int deviceId = Utils.GetMainGPU();
            yolo = new YoloSharp(new ExecutionProviderDirectML(model, deviceId));
        }
        [Fact]
        public void TestRunSegment()
        {
            string imgPath = TestDataUtils.GetImagePathSeg(TestDataUtils.Seg01);

            var res = yolo.RunSegment(imgPath);
            string ans = res.Summary();
            Assert.Equal(Yolo11.Seg01, ans);

            using Mat img = Cv2.ImRead(imgPath);
            var res2 = yolo.RunSegment(img);
            string ans2 = res2.Summary();
            Assert.Equal(Yolo11.Seg01, ans2);
        }

        [Fact]
        public void RunSegmentWithTime()
        {
            string imgPath = TestDataUtils.GetImagePathSeg(TestDataUtils.Seg01);

            var res = yolo.RunSegmentWithTime(imgPath);
            string ans = res.Items.Summary();
            Assert.Equal(Yolo11.Seg01, ans);

            using Mat img = Cv2.ImRead(imgPath);
            var res2 = yolo.RunSegmentWithTime(img);
            string ans2 = res2.Items.Summary();
            Assert.Equal(Yolo11.Seg01, ans2);
        }


        [Fact]
        public async Task TestAsyncChannel()
        {
            string imgPath = TestDataUtils.GetImagePathSeg(TestDataUtils.Seg01);

            using IYoloAsync yoloAsync = yolo.CreateAsyncChannel();

            var res = await yoloAsync.RunSegmentAsync(imgPath);
            string ans = res.Summary();
            Assert.Equal(Yolo11.Seg01, ans);

            using Mat img = Cv2.ImRead(imgPath);
            var res2 = await yoloAsync.RunSegmentAsync(img);
            string ans2 = res2.Summary();
            Assert.Equal(Yolo11.Seg01, ans2);
        }

        [Fact]
        public void TestRunBatchSegDir()
        {
            yolo.YoloConfiguration.BatchPoolSize = 4;
            var processCallback = new ProcessCallbackSeg(_dictSeg);

            string dir = TestDataUtils.GetImageDirSeg();
            var list = yolo.RunBatchSegment(dir, processCallback, ReceiveProcess);

            Assert.Equal(2, list.Length);

            foreach (var item in list)
            {
                Assert.True(_dictSeg.ContainsKey(item.ImagePath));
                Assert.Equal(_dictSeg[item.ImagePath], item.Results.Summary());
            }
        }

        [Fact]
        public void TestRunBatchSegList()
        {
            yolo.YoloConfiguration.BatchPoolSize = 4;
            var processCallback = new ProcessCallbackSeg(_dictSeg);

            List<string> imgs = TestDataUtils.GetImgSegPaths();
            var list2 = yolo.RunBatchSegment(imgs, processCallback, ReceiveProcess);

            Assert.Equal(2, list2.Length);

            foreach (var item in list2)
            {
                Assert.True(_dictSeg.ContainsKey(item.ImagePath));
                Assert.Equal(_dictSeg[item.ImagePath], item.Results.Summary());
            }
        }

        [Fact]
        public async Task RunRunBatchSegAsyncDir()
        {
            yolo.YoloConfiguration.BatchPoolSize = 4;
            var processCallback = new ProcessCallbackSeg(_dictSeg);

            string dir = TestDataUtils.GetImageDirSeg();

            var list = await yolo.RunBatchSegmentAsync(dir, processCallback, ReceiveProcess);

            Assert.Equal(2, list.Length);

            foreach (var item in list)
            {
                Assert.True(_dictSeg.ContainsKey(item.ImagePath));
                Assert.Equal(_dictSeg[item.ImagePath], item.Results.Summary());
            }
        }


        [Fact]
        public async Task RunRunBatchSegAsyncList()
        {
            yolo.YoloConfiguration.BatchPoolSize = 4;
            var processCallback = new ProcessCallbackSeg(_dictSeg);

            List<string> imgs = TestDataUtils.GetImgSegPaths();

            var list = await yolo.RunBatchSegmentAsync(imgs, processCallback, ReceiveProcess);

            Assert.Equal(2, list.Length);

            foreach (var item in list)
            {
                Assert.True(_dictSeg.ContainsKey(item.ImagePath));
                Assert.Equal(_dictSeg[item.ImagePath], item.Results.Summary());
            }
        }


        [Fact]
        public async Task BatchSegmentForeachAsync()
        {
            yolo.YoloConfiguration.BatchPoolSize = 4;
            var processCallback = new ProcessCallbackSeg(_dictSeg);

            List<string> imgs = TestDataUtils.GetImgSegPaths();

            int idx = 0;
            await foreach (var item in yolo.BatchSegmentForeachAsync(imgs))
            {
                idx++;
                Assert.True(_dictSeg.ContainsKey(item.ImagePath));
                Assert.Equal(_dictSeg[item.ImagePath], item.Results.Summary());
            }

            Assert.Equal(imgs.Count, idx);
        }


        private void ReceiveProcess(SegBatchResult e)
        {
            Assert.True(_dictSeg.ContainsKey(e.ImagePath));
            string res = e.Results.Summary();
            Assert.Equal(_dictSeg[e.ImagePath], res);
        }

        public void Dispose()
        {
            yolo.Dispose();
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
