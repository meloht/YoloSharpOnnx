using OpenCvSharp;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Providers;
using YoloSharpOnnx.TestCommon;

namespace YoloSharpOnnx.Test
{
    public class UnitTestClassifyAPI : IDisposable
    {
        private Dictionary<string, string> _dictCls;
        private string model;
        private YoloSharp yolo;

        public UnitTestClassifyAPI()
        {
            _dictCls = TestDataUtils.GetYolo11ClsDict();
            model = TestDataUtils.GetModelPath("yolo11n-cls.onnx");
            yolo = new YoloSharp(new ExecutionProviderCPU(model));
        }

        [Fact]
        public void TestRunClassify()
        {
            string imgPath = TestDataUtils.GetImagePathCls(TestDataUtils.Cls01);

            var res = yolo.RunClassify(imgPath);
            string ans = res.Summary();
            Assert.Equal(Yolo11.Cls01, ans);

            using Mat img = Cv2.ImRead(imgPath);
            var res2 = yolo.RunClassify(img);
            string ans2 = res2.Summary();
            Assert.Equal(Yolo11.Cls01, ans2);
        }

        [Fact]
        public void RunClassifyWithTime()
        {
            string imgPath = TestDataUtils.GetImagePathCls(TestDataUtils.Cls01);

            var res = yolo.RunClassifyWithTime(imgPath);
            string ans = res.Items.Summary();
            Assert.Equal(Yolo11.Cls01, ans);

            using Mat img = Cv2.ImRead(imgPath);
            var res2 = yolo.RunClassifyWithTime(img);
            string ans2 = res2.Items.Summary();
            Assert.Equal(Yolo11.Cls01, ans2);
        }


        [Fact]
        public async Task TestAsyncChannel()
        {
            string imgPath = TestDataUtils.GetImagePathCls(TestDataUtils.Cls01);

            using IYoloAsync yoloAsync = yolo.CreateAsyncChannel();

            var res = await yoloAsync.RunClassifyAsync(imgPath);
            string ans = res.Summary();
            Assert.Equal(Yolo11.Cls01, ans);

            using Mat img = Cv2.ImRead(imgPath);
            var res2 = await yoloAsync.RunClassifyAsync(img);
            string ans2 = res2.Summary();
            Assert.Equal(Yolo11.Cls01, ans2);

            await yoloAsync.CompleteAndCloseAsyncChannel();
        }

        [Fact]
        public async Task TestAsyncBatchChannel()
        {
            using IYoloAsync yoloAsync = yolo.CreateAsyncChannel();
            List<string> imgs = TestDataUtils.GetImgClsPaths();
            Dictionary<Guid, string> guidDict = new Dictionary<Guid, string>();
            int count = 0;
            foreach (var item in imgs)
            {
                using Mat img = Cv2.ImRead(item);
                Guid guid = Guid.NewGuid();
                guidDict.Add(guid, _dictCls[item]);
                await yoloAsync.RunClassifyAsync(img, guid, null, (result) =>
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
        public void TestRunBatchClsDir()
        {
            yolo.YoloConfiguration.BatchPoolSize = 4;
            var processCallback = new ProcessCallbackCls(_dictCls);

            string dir = TestDataUtils.GetImageDirCls();
            var list = yolo.RunBatchCls(dir, processCallback, ReceiveProcess);

            Assert.Equal(2, list.Length);

            foreach (var item in list)
            {
                Assert.True(_dictCls.ContainsKey(item.ImagePath));
                Assert.Equal(_dictCls[item.ImagePath], item.Results.Summary());
            }
        }

        [Fact]
        public void TestRunBatchClsList()
        {
            yolo.YoloConfiguration.BatchPoolSize = 4;
            var processCallback = new ProcessCallbackCls(_dictCls);

            List<string> imgs = TestDataUtils.GetImgClsPaths();
            var list2 = yolo.RunBatchCls(imgs, processCallback, ReceiveProcess);

            Assert.Equal(2, list2.Length);

            foreach (var item in list2)
            {
                Assert.True(_dictCls.ContainsKey(item.ImagePath));
                Assert.Equal(_dictCls[item.ImagePath], item.Results.Summary());
            }
        }

        [Fact]
        public async Task RunRunBatchClsAsyncDir()
        {
            yolo.YoloConfiguration.BatchPoolSize = 4;
            var processCallback = new ProcessCallbackCls(_dictCls);

            string dir = TestDataUtils.GetImageDirCls();

            var list = await yolo.RunBatchClsAsync(dir, processCallback, ReceiveProcess);

            Assert.Equal(2, list.Length);

            foreach (var item in list)
            {
                Assert.True(_dictCls.ContainsKey(item.ImagePath));
                Assert.Equal(_dictCls[item.ImagePath], item.Results.Summary());
            }
        }


        [Fact]
        public async Task RunRunBatchClsAsyncList()
        {
            yolo.YoloConfiguration.BatchPoolSize = 4;
            var processCallback = new ProcessCallbackCls(_dictCls);

            List<string> imgs = TestDataUtils.GetImgClsPaths();

            var list = await yolo.RunBatchClsAsync(imgs, processCallback, ReceiveProcess);

            Assert.Equal(2, list.Length);

            foreach (var item in list)
            {
                Assert.True(_dictCls.ContainsKey(item.ImagePath));
                Assert.Equal(_dictCls[item.ImagePath], item.Results.Summary());
            }
        }




        [Fact]
        public async Task BatchClsForeachAsync()
        {
            yolo.YoloConfiguration.BatchPoolSize = 4;
            var processCallback = new ProcessCallbackCls(_dictCls);

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


        private void ReceiveProcess(ClsBatchResult e)
        {
            Assert.True(_dictCls.ContainsKey(e.ImagePath));
            string res = e.Results.Summary();
            Assert.Equal(_dictCls[e.ImagePath], res);
        }

        public void Dispose()
        {
            yolo.Dispose();
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
