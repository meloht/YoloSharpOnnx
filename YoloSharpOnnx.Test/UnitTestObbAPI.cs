using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Text;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Providers;
using YoloSharpOnnx.TestCommon;

namespace YoloSharpOnnx.Test
{
    public class UnitTestObbAPI: IDisposable
    {
        private Dictionary<string, string> _dictObb;
        private string model;
        private YoloSharp yolo;

        public UnitTestObbAPI()
        {
            _dictObb = TestDataUtils.GetYolo26ObbDict();
            model = TestDataUtils.GetModelPath("yolo26n-obb.onnx");
            yolo = new YoloSharp(new ExecutionProviderCPU(model));
        }
        [Fact]
        public void TestRunObb()
        {
            string imgPath = TestDataUtils.GetImagePathObb(TestDataUtils.Obb01);

            var res = yolo.RunObbDetect(imgPath);
            string ans = res.Summary();
            Assert.Equal(Yolo26.Obb01, ans);

            using Mat img = Cv2.ImRead(imgPath);
            var res2 = yolo.RunObbDetect(img);
            string ans2 = res2.Summary();
            Assert.Equal(Yolo26.Obb01, ans2);
        }

        [Fact]
        public void RunObbWithTime()
        {
            string imgPath = TestDataUtils.GetImagePathObb(TestDataUtils.Obb01);

            var res = yolo.RunObbDetectWithTime(imgPath);
            string ans = res.Items.Summary();
            Assert.Equal(Yolo26.Obb01, ans);

            using Mat img = Cv2.ImRead(imgPath);
            var res2 = yolo.RunObbDetectWithTime(img);
            string ans2 = res2.Items.Summary();
            Assert.Equal(Yolo26.Obb01, ans2);
        }


        [Fact]
        public async Task TestAsyncChannel()
        {
            string imgPath = TestDataUtils.GetImagePathObb(TestDataUtils.Obb01);

            using IYoloAsync yoloAsync = yolo.CreateAsyncChannel();

            var res = await yoloAsync.RunObbDetectAsync(imgPath);
            string ans = res.Summary();
            Assert.Equal(Yolo26.Obb01, ans);

            using Mat img = Cv2.ImRead(imgPath);
            var res2 = await yoloAsync.RunObbDetectAsync(img);
            string ans2 = res2.Summary();
            Assert.Equal(Yolo26.Obb01, ans2);
        }

        [Fact]
        public void TestRunBatchObbDir()
        {
            yolo.YoloConfiguration.BatchPoolSize = 4;
            var processCallback = new ProcessCallbackObb(_dictObb);

            string dir = TestDataUtils.GetImageDirObb();
            var list = yolo.RunBatchObbDetect(dir, processCallback, ReceiveProcess);

            Assert.Equal(2, list.Length);

            foreach (var item in list)
            {
                Assert.True(_dictObb.ContainsKey(item.ImagePath));
                Assert.Equal(_dictObb[item.ImagePath], item.Results.Summary());
            }
        }

        [Fact]
        public void TestRunBatchObbList()
        {
            yolo.YoloConfiguration.BatchPoolSize = 4;
            var processCallback = new ProcessCallbackObb(_dictObb);

            List<string> imgs = TestDataUtils.GetImgObbPaths();
            var list2 = yolo.RunBatchObbDetect(imgs, processCallback, ReceiveProcess);

            Assert.Equal(2, list2.Length);

            foreach (var item in list2)
            {
                Assert.True(_dictObb.ContainsKey(item.ImagePath));
                Assert.Equal(_dictObb[item.ImagePath], item.Results.Summary());
            }
        }

        [Fact]
        public async Task RunRunBatchObbAsyncDir()
        {
            yolo.YoloConfiguration.BatchPoolSize = 4;
            var processCallback = new ProcessCallbackObb(_dictObb);

            string dir = TestDataUtils.GetImageDirObb();

            var list = await yolo.RunBatchObbDetectAsync(dir, processCallback, ReceiveProcess);

            Assert.Equal(2, list.Length);

            foreach (var item in list)
            {
                Assert.True(_dictObb.ContainsKey(item.ImagePath));
                Assert.Equal(_dictObb[item.ImagePath], item.Results.Summary());
            }
        }


        [Fact]
        public async Task RunRunBatchObbAsyncList()
        {
            yolo.YoloConfiguration.BatchPoolSize = 4;
            var processCallback = new ProcessCallbackObb(_dictObb);

            List<string> imgs = TestDataUtils.GetImgObbPaths();

            var list = await yolo.RunBatchObbDetectAsync(imgs, processCallback, ReceiveProcess);

            Assert.Equal(2, list.Length);

            foreach (var item in list)
            {
                Assert.True(_dictObb.ContainsKey(item.ImagePath));
                Assert.Equal(_dictObb[item.ImagePath], item.Results.Summary());
            }
        }


        [Fact]
        public async Task BatchObbForeachAsync()
        {
            yolo.YoloConfiguration.BatchPoolSize = 4;
            var processCallback = new ProcessCallbackObb(_dictObb);

            List<string> imgs = TestDataUtils.GetImgObbPaths();

            int idx = 0;
            await foreach (var item in yolo.BatchObbDetectForeachAsync(imgs))
            {
                idx++;
                Assert.True(_dictObb.ContainsKey(item.ImagePath));
                Assert.Equal(_dictObb[item.ImagePath], item.Results.Summary());
            }

            Assert.Equal(imgs.Count, idx);
        }


        private void ReceiveProcess(ObbBatchResult e)
        {
            Assert.True(_dictObb.ContainsKey(e.ImagePath));
            string res = e.Results.Summary();
            Assert.Equal(_dictObb[e.ImagePath], res);
        }

        public void Dispose()
        {
            yolo.Dispose();
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
