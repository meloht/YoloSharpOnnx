using OpenCvSharp;
using System.Threading.Channels;
using YoloSharpOnnx.Providers;

namespace YoloSharpOnnx.ConsoleDirectML
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello, World!");

            //TestChannel();
            TestBatchInfer();
            //TestInfer();

        }

        private static void TestInfer()
        {
            string modelPath = @"D:\code\model\best.onnx";
            string dir = @"D:\code\model\TestImages";

            DirectoryInfo directory = new DirectoryInfo(dir);
            var files = directory.GetFiles();

            System.Diagnostics.Stopwatch _stopwatch = new System.Diagnostics.Stopwatch();
            System.Diagnostics.Stopwatch _stopwatchTotal = new System.Diagnostics.Stopwatch();
            _stopwatchTotal.Start();
            using (YoloSharp yolo = new YoloSharp(new ExecutionProviderDirectML(modelPath, 1)))
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
            string modelPath = @"D:\code\model\best.onnx";
            string dir = @"D:\code\model\TestImages";

            DirectoryInfo directory = new DirectoryInfo(dir);
            var files = directory.GetFiles();


            using (YoloSharp yolo = new YoloSharp(new ExecutionProviderDirectML(modelPath)))
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

        }

        private static void TestBatchInfer()
        {
            string modelPath = @"D:\code\model\best.onnx";
            string dir = @"D:\code\model\TestImages";

            DirectoryInfo directory = new DirectoryInfo(dir);
            var files = directory.GetFiles();

            System.Diagnostics.Stopwatch _stopwatch = new System.Diagnostics.Stopwatch();
            _stopwatch.Start();
            using (YoloSharp yolo = new YoloSharp(new ExecutionProviderDirectML(modelPath, 1)))
            {
                yolo.BatchDetectItemCompleted += Yolo_BatchDetectCompleted;

                var list = yolo.RunBatchDetect(dir, 30);

            }
            _stopwatch.Stop();

            Console.WriteLine($"time:{_stopwatch.Elapsed}");
        }

        private static void Yolo_BatchDetectCompleted(object? sender, Models.BatchDetectionResultEventArgs e)
        {
            string ans = YoloUtils.GetResult(e.Results);
            Console.WriteLine(ans);
        }

        public static void TestChannel()
        {
            // 1. 创建 有界通道（容量=2）
            Channel<int> channel = Channel.CreateBounded<int>(new BoundedChannelOptions(100)
            {
                // 通道满时的策略：等待（默认，推荐）
                FullMode = BoundedChannelFullMode.Wait
            });

            // 生产者
            var producer = Task.Run(async () =>
            {
                for (int i = 1; i <= 500; i++)
                {
                    await channel.Writer.WriteAsync(i);
                    Console.WriteLine($"生产：{i}");
                    await Task.Delay(10);
                }
                channel.Writer.Complete();
            });

            // 消费者
            var consumer = Task.Run(async () =>
            {
                // ✅ 极简读取写法（C# 8+）
                await foreach (var msg in channel.Reader.ReadAllAsync())
                {
                    Console.WriteLine($"消费：{msg}");
                    await Task.Delay(500);
                }
            });

            Task.WaitAll(producer, consumer);
        }
    }
}
