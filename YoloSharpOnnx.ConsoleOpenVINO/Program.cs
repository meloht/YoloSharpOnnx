using YoloSharpOnnx.Providers;

namespace YoloSharpOnnx.ConsoleOpenVINO
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello, World!");
            TestInferPerf();

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

            using (YoloSharp yolo = new YoloSharp(new ExecutionProviderOpenVINO(modelPath, IntelDeviceType.GPU)))
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
