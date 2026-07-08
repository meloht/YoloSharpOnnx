using OpenCvSharp;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Providers;

namespace YoloSharpOnnx.ConsoleOpenVINO
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello, World!");
            TestInferPerf();
            //TestBatchInferPose();
            //_=TestBatchInferForeachPose();
          //  _ = TestInferBatchAsync();
            Console.ReadKey();

        }

        private static void TestInferPerf()
        {
            string modelPath = @"C:\code\model\best.onnx";
            string dir = @"C:\code\model\TestImages";

            DirectoryInfo directory = new DirectoryInfo(dir);
            var files = directory.GetFiles();
            System.Diagnostics.Stopwatch _stopwatchTotal = new System.Diagnostics.Stopwatch();
            _stopwatchTotal.Start();

            using (YoloSharp yolo = new YoloSharp(new ExecutionProviderOpenVINO(modelPath, IntelDeviceType.GPU0)))
            {
                foreach (var item in files)
                {
                    string filePath = item.Extension.ToLower();
                    if (filePath.EndsWith(".jpg") || filePath.EndsWith(".png"))
                    {

                        var res = yolo.RunDetectWithTime(item.FullName);

                        Console.WriteLine($"{res.ToString()}, {res.SpeedResult.ToString()}");
                    }
                }
            }
            _stopwatchTotal.Stop();

            Console.WriteLine($"time:{_stopwatchTotal.Elapsed}");

        }

        private static async Task TestInferBatchAsync()
        {
            string modelPath = @"D:\code\model\best.onnx";
            string dir = @"D:\code\model\TestImages";
            using var yolo = new YoloSharp(new ExecutionProviderOpenVINO(modelPath, IntelDeviceType.CPU));

            using (var yoloAsync = yolo.CreateAsyncChannel())
            {
                var files = Directory.GetFiles(dir);
                count = 1;
                for (int i = 0; i < files.Length; i++)
                {
                    using Mat img = Cv2.ImRead(files[i]);
                    await yoloAsync.RunDetectAsync(img, Guid.NewGuid(), null, ReceiveProcess);
                }
                await yoloAsync.CompleteAndCloseAsyncChannel();
            }
            
        }
        static int count = 1;
        private static void ReceiveProcess(DetectAsyncResult e)
        {
            long cost = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() - e.StartTimestamp;
            string ans = e.Results.Summary();
            Console.WriteLine($"{count++} {ans} time:{cost}ms");

        }
        private static void ReceiveProcess(PoseBatchResult e)
        {
            long cost = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() - e.StartTimestamp;
            string ans = e.Results.Summary();
            Console.WriteLine($"{count++} {ans} time:{cost}ms");

        }

        private static void TestBatchInferPose()
        {
            string modelPath = @"D:\code\model\yolo26n-pose.onnx";
            string dir = @"D:\code\model\coco8-pose";
            DirectoryInfo directory = new DirectoryInfo(dir);
            var files = directory.GetFiles();

            System.Diagnostics.Stopwatch _stopwatch = new System.Diagnostics.Stopwatch();
            int num = files.Length;
            using (YoloSharp yolo = new YoloSharp(new ExecutionProviderOpenVINO(modelPath, IntelDeviceType.CPU)))
            {

                yolo.YoloConfiguration.BatchPoolSize = 60;

                _stopwatch.Start();
                var list = yolo.RunBatchPose(dir, receiveAction: ReceiveProcess);
                _stopwatch.Stop();

            }


            Console.WriteLine($"detect {num} images, time:{_stopwatch.Elapsed}");
        }

        private static async Task TestBatchInferForeachPose()
        {
            string modelPath = @"D:\code\model\yolo26n-pose.onnx";
            string dir = @"D:\code\model\coco8-pose";
            DirectoryInfo directory = new DirectoryInfo(dir);
            var files = directory.GetFiles();

            System.Diagnostics.Stopwatch _stopwatch = new System.Diagnostics.Stopwatch();
            int num = files.Length;
            using (YoloSharp yolo = new YoloSharp(new ExecutionProviderOpenVINO(modelPath, IntelDeviceType.CPU)))
            {

                yolo.YoloConfiguration.BatchPoolSize = 60;

                _stopwatch.Start();

                await foreach (var e in yolo.BatchPoseForeachAsync(dir))
                {
                    long cost = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() - e.StartTimestamp;
                    string ans = e.Results.Summary();
                    Console.WriteLine($"{e.ImagePath} {ans} time:{cost}ms");
                }
                _stopwatch.Stop();

            }

            Console.WriteLine($"detect {num} images, time:{_stopwatch.Elapsed}");
        }
    }
}
