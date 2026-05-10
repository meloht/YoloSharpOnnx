namespace YoloSharpOnnx.TestCommon
{
    public class TestDataUtils
    {
        public const string Bus = "bus.jpg";
        public const string Zidane = "zidane.jpg";


        public static string GetImagePathDetect(string path)
        {
            return Path.Combine(AppContext.BaseDirectory, "TestData", "Images","Detect", path);
        }
        public static string GetModelPath(string path)
        {
            return Path.Combine(AppContext.BaseDirectory, "TestData", "Models", path);
        }
        public static string GetImageDirDetect()
        {
            return Path.Combine(AppContext.BaseDirectory, "TestData", "Images", "Detect");
        }

        public static Dictionary<string, string> GetYolo11Dict()
        {
            Dictionary<string, string> dict = new Dictionary<string, string>();
            dict.Add(GetImagePathDetect(Bus), Yolo11.Bus);
            dict.Add(GetImagePathDetect(Zidane), Yolo11.Zidane);

            return dict;
        }

        public static List<string> GetImgPaths()
        {
            List<string> list = [GetImagePathDetect(Bus),GetImagePathDetect(Zidane)];
            return list;
        }
    }
}
