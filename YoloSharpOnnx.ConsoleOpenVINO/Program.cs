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
            //TestInferPerf();
            _ = TestInferBatchAsync();
            Console.ReadKey();

        }

        private static void TestInferPerf()
        {
            string modelPath = @"D:\code\model\best.onnx";
            string dir = @"D:\code\model\TestImages";

            DirectoryInfo directory = new DirectoryInfo(dir);
            var files = directory.GetFiles();
            System.Diagnostics.Stopwatch _stopwatchTotal = new System.Diagnostics.Stopwatch();
            _stopwatchTotal.Start();

            using (YoloSharp yolo = new YoloSharp(new ExecutionProviderOpenVINO(modelPath, IntelDeviceType.CPU)))
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
    }
}
