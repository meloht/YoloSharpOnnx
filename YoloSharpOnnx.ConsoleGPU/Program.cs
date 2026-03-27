using YoloSharpOnnx.Providers;

namespace YoloSharpOnnx.ConsoleGPU
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello, World!");
            TestInferPerf();
            Console.WriteLine("end!");
            Console.ReadKey();
        }

        private static void TestInfer()
        {
            string modelPath = @"C:\code\model\best.onnx";
            string dir = @"C:\code\model\TestImages";

            DirectoryInfo directory = new DirectoryInfo(dir);
            var files = directory.GetFiles();
            System.Diagnostics.Stopwatch _stopwatchTotal = new System.Diagnostics.Stopwatch();
            _stopwatchTotal.Start();

            System.Diagnostics.Stopwatch _stopwatch = new System.Diagnostics.Stopwatch();
            using (YoloSharp yolo = new YoloSharp(new ExecutionProviderGPU(modelPath, 0)))
            {
                foreach (var item in files)
                {
                    string filePath = item.Extension.ToLower();
                    if (filePath.EndsWith(".jpg") || filePath.EndsWith(".png"))
                    {
                        _stopwatch.Restart();
                        var res = yolo.RunDetect(item.FullName);
                        _stopwatch.Stop();
                        string ans = YoloUtils.GetResult(res);
                        Console.WriteLine($"{ans}, time:{_stopwatch.ElapsedMilliseconds}");
                    }
                }
            }

            _stopwatchTotal.Stop();

            Console.WriteLine($"time:{_stopwatchTotal.Elapsed}");

        }
        private static void TestInferPerf()
        {
            string modelPath = @"C:\code\model\best.onnx";
            string dir = @"C:\code\model\TestImages";

            DirectoryInfo directory = new DirectoryInfo(dir);
            var files = directory.GetFiles();
            System.Diagnostics.Stopwatch _stopwatchTotal = new System.Diagnostics.Stopwatch();
            _stopwatchTotal.Start();

            using (YoloSharp yolo = new YoloSharp(new ExecutionProviderGPU(modelPath, 0)))
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
    }
}
