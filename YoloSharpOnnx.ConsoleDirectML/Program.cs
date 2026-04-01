using OpenCvSharp;
using System.Runtime.Intrinsics.X86;
using System.Threading.Channels;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Providers;

namespace YoloSharpOnnx.ConsoleDirectML
{
    internal class Program
    {
        static int _deviceId = 0;
        static string modelPath = @"D:\code\model\best.onnx";
        static string dir = @"D:\code\model\TestImages";
        static void Main(string[] args)
        {
            Console.WriteLine("Hello, World!");

            //TestChannel();
            //TestBatchInfer();
            // TestInferPerf();
            //TestInfer();
            //_ = Task.Run(async () => await TestInferAsync());
            _ = TestBatchForeachInfer();
            Console.WriteLine("end!");
            Console.ReadKey();

        }

        private static void TestInfer()
        {
            DirectoryInfo directory = new DirectoryInfo(dir);
            var files = directory.GetFiles();

            System.Diagnostics.Stopwatch _stopwatch = new System.Diagnostics.Stopwatch();
            System.Diagnostics.Stopwatch _stopwatchTotal = new System.Diagnostics.Stopwatch();
            _stopwatchTotal.Start();
            using (YoloSharp yolo = new YoloSharp(new ExecutionProviderDirectML(modelPath, _deviceId)))
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

            DirectoryInfo directory = new DirectoryInfo(dir);
            var files = directory.GetFiles();
            System.Diagnostics.Stopwatch _stopwatchTotal = new System.Diagnostics.Stopwatch();
            _stopwatchTotal.Start();

            long totalInfer = 0;
            int count = 0;
            using (YoloSharp yolo = new YoloSharp(new ExecutionProviderDirectML(modelPath, _deviceId)))
            {
                foreach (var item in files)
                {
                    string filePath = item.Extension.ToLower();
                    if (filePath.EndsWith(".jpg") || filePath.EndsWith(".png"))
                    {
                        count++;
                        var res = yolo.RunDetectWithTime(item.FullName);
                        totalInfer += res.SpeedResult.Inference;
                        Console.WriteLine($"{res.ToString()}, {res.SpeedResult.ToString()}");
                    }
                }
            }

            _stopwatchTotal.Stop();

            float avg = totalInfer / (float)count;
            Console.WriteLine($"total time:{_stopwatchTotal.Elapsed},count:{count} Infer avg time:{avg}ms");

        }
        private static async Task TestInferAsync()
        {

            using var yolo = new YoloSharp(new ExecutionProviderDirectML(modelPath, _deviceId));
            System.Diagnostics.Stopwatch _stopwatchTotal = new System.Diagnostics.Stopwatch();
            _stopwatchTotal.Start();
            var files = Directory.GetFiles(dir);
            yolo.YoloConfiguration.BatchPoolSize = 5;
            using (var yoloAsync = yolo.CreateAsyncChannel())
            {
                for (int i = 0; i < files.Length; i++)
                {
                   
                    var res = await yoloAsync.RunDetectAsync(files[i]);
                    Console.WriteLine($"{i + 1} {YoloUtils.GetResult(res)}");
                }

            }

            _stopwatchTotal.Stop();
            var avg = _stopwatchTotal.ElapsedMilliseconds / files.Length;
            Console.WriteLine($"total time:{_stopwatchTotal.Elapsed}, count:{files.Length} Infer avg time:{avg}ms");

        }
        private static void TestBatchInfer()
        {

            DirectoryInfo directory = new DirectoryInfo(dir);
            var files = directory.GetFiles();

            System.Diagnostics.Stopwatch _stopwatch = new System.Diagnostics.Stopwatch();
            _stopwatch.Start();
            int num = files.Length;
            using (YoloSharp yolo = new YoloSharp(new ExecutionProviderDirectML(modelPath, _deviceId)))
            {
                yolo.YoloConfiguration.BatchPoolSize = 30;
                yolo.BatchDetectItemCompleted += Yolo_BatchDetectCompleted;

                var list = yolo.RunBatchDetect(dir, new ProcessCallback(), ReceiveProcess);

            }
            _stopwatch.Stop();

            Console.WriteLine($"detect {num} images, time:{_stopwatch.Elapsed}");
        }

        private static async Task TestBatchForeachInfer()
        {
            var files = Directory.GetFiles(dir);
            System.Diagnostics.Stopwatch _stopwatch = new System.Diagnostics.Stopwatch();
            _stopwatch.Start();
            int num = files.Length;
            using (YoloSharp yolo = new YoloSharp(new ExecutionProviderDirectML(modelPath, _deviceId)))
            {
                yolo.YoloConfiguration.BatchPoolSize = 30;
                yolo.BatchDetectItemCompleted += Yolo_BatchDetectCompleted;

                await foreach (var item in yolo.BatchDetectForeachAsync(files.ToList()))
                {
                    Console.WriteLine($"{item.ImagePath} {YoloUtils.GetResult(item.Results)}");
                }

            }
            _stopwatch.Stop();

            Console.WriteLine($"detect {num} images, time:{_stopwatch.Elapsed}");
        }

        private static void Yolo_BatchDetectCompleted(object? sender, DetectionBatchResult e)
        {
            long cost = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() - e.StartTimestamp;
            string ans = YoloUtils.GetResult(e.Results);
            Console.WriteLine($"{ans} time:{cost}ms");
        }

        private static void ReceiveProcess(DetectionBatchResult e)
        {

            string res = YoloUtils.GetResult(e.Results);

        }
        internal class ProcessCallback : IBatchProcessCallback
        {

            public void ReceiveProcessResult(DetectionBatchResult e)
            {

                string res = YoloUtils.GetResult(e.Results);

            }

        }

        public static async Task TestChannel()
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
                for (int i = 1; i <= 100; i++)
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
                // 极简读取写法（C# 8+）
                await foreach (var msg in channel.Reader.ReadAllAsync())
                {
                    Console.WriteLine($"消费：{msg}");
                    await Task.Delay(12);
                }
            });

            Task.WaitAll(consumer, producer);


        }
    }
}
