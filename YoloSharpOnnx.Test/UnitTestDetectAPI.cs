using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Text;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Providers;
using YoloSharpOnnx.TestCommon;

namespace YoloSharpOnnx.Test
{
    public class UnitTestDetectAPI : IDisposable
    {
        private Dictionary<string, string> _dict;

        private string model;
        private YoloSharp yolo;
        public UnitTestDetectAPI()
        {
            _dict = TestDataUtils.GetYolo11Dict();
            model = TestDataUtils.GetModelPath("yolo11n.onnx");
            yolo = new YoloSharp(new ExecutionProviderCPU(model));
        }

        [Fact]
        public void TestRunDetect()
        {
            string imgPath = TestDataUtils.GetImagePathDetect(TestDataUtils.Bus);

            var res = yolo.RunDetect(imgPath);
            string ans = res.SummaryOrder();
            Assert.Equal(Yolo11.Bus, ans);

            using Mat img = Cv2.ImRead(imgPath);
            var res2 = yolo.RunDetect(img);
            string ans2 = res2.SummaryOrder();
            Assert.Equal(Yolo11.Bus, ans2);
        }

        [Fact]
        public void TestRunDetectWithTime()
        {
            string imgPath = TestDataUtils.GetImagePathDetect(TestDataUtils.Bus);

            var res = yolo.RunDetectWithTime(imgPath);
            string ans = res.Items.SummaryOrder();
            Assert.Equal(Yolo11.Bus, ans);

            using Mat img = Cv2.ImRead(imgPath);
            var res2 = yolo.RunDetectWithTime(img);
            string ans2 = res2.Items.SummaryOrder();
            Assert.Equal(Yolo11.Bus, ans2);
        }

        [Fact]
        public async Task TestAsyncChannel()
        {
            string imgPath = TestDataUtils.GetImagePathDetect(TestDataUtils.Bus);

            using IYoloAsync yoloAsync = yolo.CreateAsyncChannel();

            var res = await yoloAsync.RunDetectAsync(imgPath);
            string ans = res.SummaryOrder();
            Assert.Equal(Yolo11.Bus, ans);

            using Mat img = Cv2.ImRead(imgPath);
            var res2 = await yoloAsync.RunDetectAsync(img);
            string ans2 = res2.SummaryOrder();
            Assert.Equal(Yolo11.Bus, ans2);

            await yoloAsync.CompleteAndCloseAsyncChannel();
        }
        [Fact]
        public async Task TestAsyncBatchChannel()
        {
            using IYoloAsync yoloAsync = yolo.CreateAsyncChannel();
            List<string> imgs = TestDataUtils.GetImgPaths();
            Dictionary<Guid, string> guidDict = new Dictionary<Guid, string>();
            int count = 0;
            foreach (var item in imgs)
            {
                using Mat img = Cv2.ImRead(item);
                Guid guid = Guid.NewGuid();
                guidDict.Add(guid, _dict[item]);
                await yoloAsync.RunDetectAsync(img, guid, null, (result) =>
                {
                    Assert.True(guidDict.ContainsKey(result.Guid));
                    Assert.Equal(guidDict[result.Guid], result.Results.Summary());
                    count++;
                });
            }
            await yoloAsync.CompleteAndCloseAsyncChannel();
            Assert.Equal(imgs.Count, count);
        }

        [Fact]
        public void TestBatchDetectDir()
        {
            yolo.YoloConfiguration.BatchPoolSize = 4;
            var processCallback = new ProcessCallback(_dict);

            string dir = TestDataUtils.GetImageDirDetect();
            var list = yolo.RunBatchDetect(dir, processCallback, ReceiveProcess);

            Assert.Equal(2, list.Length);

            foreach (var item in list)
            {
                Assert.True(_dict.ContainsKey(item.ImagePath));
                Assert.Equal(_dict[item.ImagePath], item.Results.SummaryOrder());
            }
        }

        [Fact]
        public void TestBatchDetectList()
        {
            yolo.YoloConfiguration.BatchPoolSize = 4;
            var processCallback = new ProcessCallback(_dict);

            List<string> imgs = TestDataUtils.GetImgPaths();
            var list2 = yolo.RunBatchDetect(imgs, processCallback, ReceiveProcess);

            Assert.Equal(2, list2.Length);

            foreach (var item in list2)
            {
                Assert.True(_dict.ContainsKey(item.ImagePath));
                Assert.Equal(_dict[item.ImagePath], item.Results.SummaryOrder());
            }
        }

        [Fact]
        public async Task RunBatchDetectAsyncDir()
        {
            yolo.YoloConfiguration.BatchPoolSize = 4;
            var processCallback = new ProcessCallback(_dict);

            string dir = TestDataUtils.GetImageDirDetect();

            var list = await yolo.RunBatchDetectAsync(dir, processCallback, ReceiveProcess);

            Assert.Equal(2, list.Length);

            foreach (var item in list)
            {
                Assert.True(_dict.ContainsKey(item.ImagePath));
                Assert.Equal(_dict[item.ImagePath], item.Results.SummaryOrder());
            }
        }

        [Fact]
        public async Task RunBatchDetectAsyncList()
        {
            yolo.YoloConfiguration.BatchPoolSize = 4;
            var processCallback = new ProcessCallback(_dict);

            List<string> imgs = TestDataUtils.GetImgPaths();

            var list = await yolo.RunBatchDetectAsync(imgs, processCallback, ReceiveProcess);

            Assert.Equal(2, list.Length);

            foreach (var item in list)
            {
                Assert.True(_dict.ContainsKey(item.ImagePath));
                Assert.Equal(_dict[item.ImagePath], item.Results.SummaryOrder());
            }
        }

        [Fact]
        public async Task BatchDetectForeachAsync()
        {
            yolo.YoloConfiguration.BatchPoolSize = 4;
            var processCallback = new ProcessCallback(_dict);

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

        private void ReceiveProcess(DetectionBatchResult e)
        {
            Assert.True(_dict.ContainsKey(e.ImagePath));
            string res = e.Results.SummaryOrder();
            Assert.Equal(_dict[e.ImagePath], res);
        }

        public void Dispose()
        {
            yolo.Dispose();
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
