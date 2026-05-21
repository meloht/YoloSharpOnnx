using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Text;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Providers;
using YoloSharpOnnx.TestCommon;

namespace YoloSharpOnnx.TestIoBinding
{
    public class UnitTestYoloPose : IDisposable
    {
        private Dictionary<string, string> _dictPose;

        private YoloSharp yolo11n;
        private YoloSharp yolo8n;
        private YoloSharp yolo26n;
        private int deviceId;
        public UnitTestYoloPose()
        {
            _dictPose = TestDataUtils.GetYolo11PoseDict();
            deviceId = Utils.GetMainGPU();
            yolo11n = new YoloSharp(new ExecutionProviderDirectML(TestDataUtils.GetModelPath("yolo11n-pose.onnx"), deviceId));
            yolo8n = new YoloSharp(new ExecutionProviderDirectML(TestDataUtils.GetModelPath("yolov8n-pose.onnx"), deviceId));
            yolo26n = new YoloSharp(new ExecutionProviderDirectML(TestDataUtils.GetModelPath("yolo26n-pose.onnx"), deviceId));
        }

        [Theory]
        [InlineData(TestDataUtils.Pose01, Yolo11.Pose01)]
        [InlineData(TestDataUtils.Pose02, Yolo11.Pose02)]
        public void TestPoseYolo11(string path, string boxs)
        {
            string imgPath = TestDataUtils.GetImagePathPose(path);

            var res = yolo11n.RunPose(imgPath);
            string ans = res.Summary();
            Assert.Equal(boxs, ans);

            var res2 = yolo11n.RunPoseWithTime(imgPath);
            string ans2 = res2.Items.Summary();
            Assert.Equal(boxs, ans2);
        }

        [Theory]
        [InlineData(TestDataUtils.Pose01, Yolo8.Pose01)]
        [InlineData(TestDataUtils.Pose02, Yolo8.Pose02)]
        public void TestPoseYolo8(string path, string boxs)
        {
            string imgPath = TestDataUtils.GetImagePathPose(path);

            var res = yolo8n.RunPose(imgPath);
            string ans = res.Summary();
            Assert.Equal(boxs, ans);

            var res2 = yolo8n.RunPoseWithTime(imgPath);
            string ans2 = res2.Items.Summary();
            Assert.Equal(boxs, ans2);
        }

        [Theory]
        [InlineData(TestDataUtils.Pose01, Yolo26.Pose01)]
        [InlineData(TestDataUtils.Pose02, Yolo26.Pose02)]
        public void TestPoseYolo26(string path, string boxs)
        {
            string imgPath = TestDataUtils.GetImagePathPose(path);

            var res = yolo26n.RunPose(imgPath);
            string ans = res.Summary();
            Assert.Equal(boxs, ans);

            var res2 = yolo26n.RunPoseWithTime(imgPath);
            string ans2 = res2.Items.Summary();
            Assert.Equal(boxs, ans2);
        }

        [Fact]
        public async Task TestPoseAsyncYolo11()
        {

            string model = TestDataUtils.GetModelPath("yolo11n-pose.onnx");
            using YoloSharp yolo = new YoloSharp(new ExecutionProviderDirectML(model, deviceId));
            using var yoloAsync = yolo.CreateAsyncChannel();

            foreach (var item in _dictPose)
            {
                var res = await yoloAsync.RunPoseAsync(item.Key);
                Assert.Equal(item.Value, res.Summary());
            }
            foreach (var item in _dictPose)
            {
                using var img = Cv2.ImRead(item.Key);
                var res = await yoloAsync.RunPoseAsync(img);
                Assert.Equal(item.Value, res.Summary());
            }
        }

        [Fact]
        public async Task TestPoseBatchForeachAsync()
        {
            yolo11n.YoloConfiguration.BatchPoolSize = 4;

            List<string> imgs = TestDataUtils.GetImgPosePaths();
            int idx = 0;
            await foreach (var item in yolo11n.BatchPoseForeachAsync(imgs))
            {
                idx++;
                Assert.True(_dictPose.ContainsKey(item.ImagePath));
                Assert.Equal(_dictPose[item.ImagePath], item.Results.Summary());
            }

            Assert.Equal(imgs.Count, idx);
        }

        [Fact]
        public void TestPoseBatch()
        {
            string dir = TestDataUtils.GetImageDirPose();

            yolo11n.YoloConfiguration.BatchPoolSize = 4;
            var processCallback = new ProcessCallbackPose(_dictPose);
            var list = yolo11n.RunBatchPose(dir, processCallback, ReceiveProcess);

            Assert.Equal(2, list.Length);

            foreach (var item in list)
            {
                Assert.True(_dictPose.ContainsKey(item.ImagePath));
                Assert.Equal(_dictPose[item.ImagePath], item.Results.Summary());
            }

        }

        private void ReceiveProcess(PoseBatchResult e)
        {
            Assert.True(_dictPose.ContainsKey(e.ImagePath));
            string res = e.Results.Summary();
            Assert.Equal(_dictPose[e.ImagePath], res);
        }

        public void Dispose()
        {
            yolo11n.Dispose();
            yolo26n.Dispose();
            yolo8n.Dispose();
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
