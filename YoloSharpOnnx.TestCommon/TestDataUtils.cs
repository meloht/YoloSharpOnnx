namespace YoloSharpOnnx.TestCommon
{
    public class TestDataUtils
    {
        public const string Bus = "bus.jpg";
        public const string Zidane = "zidane.jpg";


        public static string GetImagePath(string path)
        {
            return Path.Combine(AppContext.BaseDirectory, "TestData", "Images", path);
        }
        public static string GetModelPath(string path)
        {
            return Path.Combine(AppContext.BaseDirectory, "TestData", "Models", path);
        }
        public static string GetImageDir()
        {
            return Path.Combine(AppContext.BaseDirectory, "TestData", "Images");
        }

        public static Dictionary<string, string> GetYolo11Dict()
        {
            Dictionary<string, string> dict = new Dictionary<string, string>();
            dict.Add(GetImagePath(Bus), Yolo11.Bus);
            dict.Add(GetImagePath(Zidane), Yolo11.Zidane);

            return dict;
        }
    }
}
