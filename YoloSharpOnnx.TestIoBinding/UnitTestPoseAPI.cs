using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Text;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Providers;
using YoloSharpOnnx.TestCommon;

namespace YoloSharpOnnx.TestIoBinding
{
    public class UnitTestPoseAPI : IDisposable
    {
        private Dictionary<string, string> _dictPose;
        private string model;
        private YoloSharp yolo;

        public UnitTestPoseAPI()
        {
            _dictPose = TestDataUtils.GetYolo11PoseDict();
            model = TestDataUtils.GetModelPath("yolo11n-pose.onnx");
            int deviceId = Utils.GetMainGPU();
            yolo = new YoloSharp(new ExecutionProviderDirectML(model, deviceId));
        }
        [Fact]
        public void TestRunPose()
        {
            string imgPath = TestDataUtils.GetImagePathPose(TestDataUtils.Pose01);

            var res = yolo.RunPose(imgPath);
            string ans = res.Summary();
            Assert.Equal(Yolo11.Pose01, ans);

            using Mat img = Cv2.ImRead(imgPath);
            var res2 = yolo.RunPose(img);
            string ans2 = res2.Summary();
            Assert.Equal(Yolo11.Pose01, ans2);
        }

        [Fact]
        public void RunPoseWithTime()
        {
            string imgPath = TestDataUtils.GetImagePathPose(TestDataUtils.Pose01);

            var res = yolo.RunPoseWithTime(imgPath);
            string ans = res.Items.Summary();
            Assert.Equal(Yolo11.Pose01, ans);

            using Mat img = Cv2.ImRead(imgPath);
            var res2 = yolo.RunPoseWithTime(img);
            string ans2 = res2.Items.Summary();
            Assert.Equal(Yolo11.Pose01, ans2);
        }


        [Fact]
        public async Task TestAsyncChannel()
        {
            string imgPath = TestDataUtils.GetImagePathPose(TestDataUtils.Pose01);

            using IYoloAsync yoloAsync = yolo.CreateAsyncChannel();

            var res = await yoloAsync.RunPoseAsync(imgPath);
            string ans = res.Summary();
            Assert.Equal(Yolo11.Pose01, ans);

            using Mat img = Cv2.ImRead(imgPath);
            var res2 = await yoloAsync.RunPoseAsync(img);
            string ans2 = res2.Summary();
            Assert.Equal(Yolo11.Pose01, ans2);
            await yoloAsync.CompleteAndCloseAsyncChannel();
        }


        [Fact]
        public async Task TestAsyncBatchChannel()
        {
            using IYoloAsync yoloAsync = yolo.CreateAsyncChannel();
            List<string> imgs = TestDataUtils.GetImgPosePaths();
            Dictionary<Guid, string> guidDict = new Dictionary<Guid, string>();
            int count = 0;
            foreach (var item in imgs)
            {
                using Mat img = Cv2.ImRead(item);
                Guid guid = Guid.NewGuid();
                guidDict.Add(guid, _dictPose[item]);
                await yoloAsync.RunPoseAsync(img, guid, null, (result) =>
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
        public async Task TestAsyncBatchImagePathChannel()
        {
            using IYoloAsync yoloAsync = yolo.CreateAsyncChannel();
            List<string> imgs = TestDataUtils.GetImgPosePaths();
            Dictionary<Guid, string> guidDict = new Dictionary<Guid, string>();
            int count = 0;
            foreach (var item in imgs)
            {
                Guid guid = Guid.NewGuid();
                guidDict.Add(guid, _dictPose[item]);
                await yoloAsync.RunPoseAsync(item, guid, null, (result) =>
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
        public void TestRunBatchPoseDir()
        {
            yolo.YoloConfiguration.BatchPoolSize = 4;
            var processCallback = new ProcessCallbackPose(_dictPose);

            string dir = TestDataUtils.GetImageDirPose();
            var list = yolo.RunBatchPose(dir, processCallback, ReceiveProcess);

            Assert.Equal(2, list.Length);

            foreach (var item in list)
            {
                Assert.True(_dictPose.ContainsKey(item.ImagePath));
                Assert.Equal(_dictPose[item.ImagePath], item.Results.Summary());
            }
        }

        [Fact]
        public void TestRunBatchPoseList()
        {
            yolo.YoloConfiguration.BatchPoolSize = 4;
            var processCallback = new ProcessCallbackPose(_dictPose);

            List<string> imgs = TestDataUtils.GetImgPosePaths();
            var list2 = yolo.RunBatchPose(imgs, processCallback, ReceiveProcess);

            Assert.Equal(2, list2.Length);

            foreach (var item in list2)
            {
                Assert.True(_dictPose.ContainsKey(item.ImagePath));
                Assert.Equal(_dictPose[item.ImagePath], item.Results.Summary());
            }
        }

        [Fact]
        public async Task RunRunBatchPoseAsyncDir()
        {
            yolo.YoloConfiguration.BatchPoolSize = 4;
            var processCallback = new ProcessCallbackPose(_dictPose);

            string dir = TestDataUtils.GetImageDirPose();

            var list = await yolo.RunBatchPoseAsync(dir, processCallback, ReceiveProcess);

            Assert.Equal(2, list.Length);

            foreach (var item in list)
            {
                Assert.True(_dictPose.ContainsKey(item.ImagePath));
                Assert.Equal(_dictPose[item.ImagePath], item.Results.Summary());
            }
        }


        [Fact]
        public async Task RunRunBatchPoseAsyncList()
        {
            yolo.YoloConfiguration.BatchPoolSize = 4;
            var processCallback = new ProcessCallbackPose(_dictPose);

            List<string> imgs = TestDataUtils.GetImgPosePaths();

            var list = await yolo.RunBatchPoseAsync(imgs, processCallback, ReceiveProcess);

            Assert.Equal(2, list.Length);

            foreach (var item in list)
            {
                Assert.True(_dictPose.ContainsKey(item.ImagePath));
                Assert.Equal(_dictPose[item.ImagePath], item.Results.Summary());
            }
        }


        [Fact]
        public async Task BatchPoseForeachAsync()
        {
            yolo.YoloConfiguration.BatchPoolSize = 4;
            var processCallback = new ProcessCallbackPose(_dictPose);

            List<string> imgs = TestDataUtils.GetImgPosePaths();

            int idx = 0;
            await foreach (var item in yolo.BatchPoseForeachAsync(imgs))
            {
                idx++;
                Assert.True(_dictPose.ContainsKey(item.ImagePath));
                Assert.Equal(_dictPose[item.ImagePath], item.Results.Summary());
            }

            Assert.Equal(imgs.Count, idx);
        }

        [Fact]
        public async Task BatchPoseForeachDirAsync()
        {
            yolo.YoloConfiguration.BatchPoolSize = 4;
            var processCallback = new ProcessCallbackPose(_dictPose);

            string dir = TestDataUtils.GetImageDirPose();
            List<string> imgs = TestDataUtils.GetImgPosePaths();

            int idx = 0;
            await foreach (var item in yolo.BatchPoseForeachAsync(dir))
            {
                idx++;
                Assert.True(_dictPose.ContainsKey(item.ImagePath));
                Assert.Equal(_dictPose[item.ImagePath], item.Results.Summary());
            }

            Assert.Equal(imgs.Count, idx);
        }

        private void ReceiveProcess(PoseBatchResult e)
        {
            Assert.True(_dictPose.ContainsKey(e.ImagePath));
            string res = e.Results.Summary();
            Assert.Equal(_dictPose[e.ImagePath], res);
        }

        public void Dispose()
        {
            yolo.Dispose();
        }

        internal class ProcessCallbackPose : IBatchProcessCallback<PoseBatchResult>
        {
            private Dictionary<string, string> _dict;
            public ProcessCallbackPose(Dictionary<string, string> dict)
            {
                _dict = dict;
            }
            public void ReceiveProcessResult(PoseBatchResult e)
            {
                Assert.True(_dict.ContainsKey(e.ImagePath));
                string res = e.Results.Summary();
                Assert.Equal(_dict[e.ImagePath], res);
            }

        }
    }
}
