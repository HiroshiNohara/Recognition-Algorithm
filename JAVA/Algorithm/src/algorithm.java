import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;

public class algorithm {
	public Mat lbp(Mat src) {
		Size size = new Size(src.rows() - 2, src.cols() - 2);
		Mat dst = Mat.zeros(size, CvType.CV_8UC1);
		for (int i = 1; i < src.rows() - 1; i++) {
			for (int j = 1; j < src.cols() - 1; j++) {
				double[] center = src.get(i, j);
				char code = 0;
				code |= (src.get(i - 1, j - 1)[0] >= center[0] ? 1 : 0) << 7;
				code |= (src.get(i - 1, j)[0] >= center[0] ? 1 : 0) << 6;
				code |= (src.get(i - 1, j + 1)[0] >= center[0] ? 1 : 0) << 5;
				code |= (src.get(i, j + 1)[0] >= center[0] ? 1 : 0) << 4;
				code |= (src.get(i + 1, j + 1)[0] >= center[0] ? 1 : 0) << 3;
				code |= (src.get(i + 1, j)[0] >= center[0] ? 1 : 0) << 2;
				code |= (src.get(i + 1, j - 1)[0] >= center[0] ? 1 : 0) << 1;
				code |= (src.get(i, j - 1)[0] >= center[0] ? 1 : 0) << 0;
				dst.put(i - 1, j - 1, code);
			}
		}
		return dst;
	}
	
	public Mat elbp(Mat src, int radius, int neighbors) {
		Size size = new Size(src.rows() - 2 * radius, src.cols() - 2 * radius);
		Mat dst = Mat.zeros(size, CvType.CV_8UC1);
		for (int n = 0; n < neighbors; n++) {
			float x = (float)(radius * Math.cos(2.0 * Math.PI * n / (float)neighbors));
			float y = (float)(-radius * Math.sin(2.0 * Math.PI * n / (float)neighbors));
			int fx = (int) (Math.floor(x));
			int fy = (int) (Math.floor(y));
			int cx = (int) (Math.ceil(x));
			int cy = (int) (Math.ceil(y));
			float ty = y - fy;
			float tx = x - fx;
			float w1 = (1 - tx) * (1 - ty);
			float w2 = tx * (1 - ty);
			float w3 = (1 - tx) * ty;
			float w4 = tx * ty;
			for (int i = radius; i < src.rows() - radius; i++) {
				for (int j = radius; j < src.cols() - radius; j++) {
					float t = (float)(w1 * src.get(i + fy, j + fx)[0] + w2 * src.get(i + fy, j + cx)[0] 
							+ w3 * src.get(i + cy, j + fx)[0] + w4 * src.get(i + cy, j + cx)[0]);
					dst.put(i - radius, j - radius, (int) dst.get(i - radius, j - radius)[0] + (t >= src.get(i, j)[0] ? 1 : 0) * (int) Math.pow(2, n));
				}
			}
		}
		return dst;
	}
	
	public Mat DCP1(Mat src, int Rin, int Rex){
		int neighbors = 4;
		Size size = new Size(src.rows() - 2 * Rex, src.cols() - 2 * Rex);
		Mat dst = Mat.zeros(size, CvType.CV_8UC1);
		float pixelA, pixelB;
		for (int k = 0; k < neighbors; k++){
			float Ax = (float)(Rin * Math.sin(Math.PI * (2 * k) / (float)neighbors));
			float Ay = (float)(Rin * Math.cos(Math.PI * (2 * k) / (float)neighbors));
			float Bx = (float)(Rex * Math.sin(Math.PI * (2 * k) / (float)neighbors));
			float By = (float)(Rex * Math.cos(Math.PI * (2 * k) / (float)neighbors));
			int fAx = (int)(Math.floor(Ax));
			int fAy = (int)(Math.floor(Ay));
			int cAx = (int)(Math.ceil(Ax));
			int cAy = (int)(Math.ceil(Ay));
			float tAy = Ay - fAy;
			float tAx = Ax - fAx;
			float wA1 = (1 - tAx) * (1 - tAy);
			float wA2 = tAx * (1 - tAy);
			float wA3 = (1 - tAx) * tAy;
			float wA4 = tAx * tAy;

			int fBx = (int)(Math.floor(Bx));
			int fBy = (int)(Math.floor(By));
			int cBx = (int)(Math.ceil(Bx));
			int cBy = (int)(Math.ceil(By));
			float tBy = By - fBy;
			float tBx = Bx - fBx;
			float wB1 = (1 - tBx) * (1 - tBy);
			float wB2 = tBx * (1 - tBy);
			float wB3 = (1 - tBx) * tBy;
			float wB4 = tBx * tBy;
			for (int i = Rex; i < src.rows() - Rex; i++){
				for (int j = Rex; j < src.cols() - Rex; j++){
					float pixelO = (float)src.get(i, j)[0];
					pixelA = (float)(src.get(i + fAy, j + fAx)[0] * wA1 + src.get(i + fAy, j + cAx)[0] *wA2 + src.get(i + cAy, j + fAx)[0] * wA3 + src.get(i + cAy, j + cAx)[0] *wA4);
					pixelB = (float)(src.get(i + fBy, j + fBx)[0] * wB1 + src.get(i + fBy, j + cBx)[0] *wB2 + src.get(i + cBy, j + fBx)[0] * wB3 + src.get(i + cBy, j + cBx)[0] *wB4);
					dst.put(i - Rex, j - Rex , (int)dst.get(i - Rex, j - Rex)[0] + ((pixelA >= pixelO ? 1 : 0) * 2 + (pixelB >= pixelA ? 1 : 0)) * (int)Math.pow(4, k));
				}
			}
		}
		return dst;
	}

	public Mat DCP2(Mat src, int Rin, int Rex){
		int neighbors = 4;
		Size size = new Size(src.rows() - 2 * Rex, src.cols() - 2 * Rex);
		Mat dst = Mat.zeros(size, CvType.CV_8UC1);
		float pixelA, pixelB;
		for (int k = 0; k < neighbors; k++){
			float Ax = (float)(Rin * Math.sin(Math.PI * (2 * k + 1) / (float)neighbors));
			float Ay = (float)(Rin * Math.cos(Math.PI * (2 * k + 1) / (float)neighbors));
			float Bx = (float)(Rex * Math.sin(Math.PI * (2 * k + 1) / (float)neighbors));
			float By = (float)(Rex * Math.cos(Math.PI * (2 * k + 1) / (float)neighbors));
			int fAx = (int)(Math.floor(Ax));
			int fAy = (int)(Math.floor(Ay));
			int cAx = (int)(Math.ceil(Ax));
			int cAy = (int)(Math.ceil(Ay));
			float tAy = Ay - fAy;
			float tAx = Ax - fAx;
			float wA1 = (1 - tAx) * (1 - tAy);
			float wA2 = tAx * (1 - tAy);
			float wA3 = (1 - tAx) * tAy;
			float wA4 = tAx * tAy;

			int fBx = (int)(Math.floor(Bx));
			int fBy = (int)(Math.floor(By));
			int cBx = (int)(Math.ceil(Bx));
			int cBy = (int)(Math.ceil(By));
			float tBy = By - fBy;
			float tBx = Bx - fBx;
			float wB1 = (1 - tBx) * (1 - tBy);
			float wB2 = tBx * (1 - tBy);
			float wB3 = (1 - tBx) * tBy;
			float wB4 = tBx * tBy;
			for (int i = Rex; i < src.rows() - Rex; i++){
				for (int j = Rex; j < src.cols() - Rex; j++){
					float pixelO = (float)src.get(i, j)[0];
					pixelA = (float)(src.get(i + fAy, j + fAx)[0] * wA1 + src.get(i + fAy, j + cAx)[0] *wA2 + src.get(i + cAy, j + fAx)[0] * wA3 + src.get(i + cAy, j + cAx)[0] *wA4);
					pixelB = (float)(src.get(i + fBy, j + fBx)[0] * wB1 + src.get(i + fBy, j + cBx)[0] *wB2 + src.get(i + cBy, j + fBx)[0] * wB3 + src.get(i + cBy, j + cBx)[0] *wB4);
					dst.put(i - Rex, j - Rex , (int)dst.get(i - Rex, j - Rex)[0] + ((pixelA >= pixelO ? 1 : 0) * 2 + (pixelB >= pixelA ? 1 : 0)) * (int)Math.pow(4, k));
				}
			}
		}
		return dst;
	}
	
	public Mat LTP1(Mat src, int radius, int neighbors, float threshold, boolean adaption)
	{
		Size size = new Size(src.rows() - 2 * radius, src.cols() - 2 * radius);
		Mat dst = Mat.zeros(size, CvType.CV_8UC1);
		float[] pixel_array = new float[neighbors];
		float sum;
		for (int i = radius; i < src.rows() - radius; i++) {
			for (int j = radius; j < src.cols() - radius; j++) {
				float center = (float)src.get(i, j)[0];
				sum = 0;
				for (int n = 0; n < neighbors; n++) {
					float x = i + (float)(radius * Math.cos(2.0 * Math.PI * n / (float)neighbors));
					float y = j - (float)(radius * Math.sin(2.0 * Math.PI * n / (float)neighbors));
					int fx = (int)(Math.floor(x));
					int fy = (int)(Math.floor(y));
					int cx = (int)(Math.ceil(x));
					int cy = (int)(Math.ceil(y));
					float ty = y - fy;
					float tx = x - fx;
					float w1 = (1 - tx) * (1 - ty);
					float w2 = tx * (1 - ty);
					float w3 = (1 - tx) * ty;
					float w4 = tx * ty;
					float t = (float)(w1 * src.get(fx, fy)[0] + w2 * src.get(cx, fy)[0] + w3 * src.get(fx, cy)[0] + w4 * src.get(cx, cy)[0]);
					sum += Math.pow(t - center, 2);
					pixel_array[n] = t;
				}
				float thre = adaption == true ? 0.003f * sum : threshold; //0.003 is the variance threshold
				for (int k = 0; k < neighbors; k++) {
					dst.put(i - radius, j - radius, (int)dst.get(i - radius, j - radius)[0] + Math.pow(2, k) * (pixel_array[k] - src.get(i, j)[0] >= thre ? 1 : 0));
				}
			}
		}
		return dst;
	}

	public Mat LTP2(Mat src, int radius, int neighbors, float threshold, boolean adaption)
	{
		Size size = new Size(src.rows() - 2 * radius, src.cols() - 2 * radius);
		Mat dst = Mat.zeros(size, CvType.CV_8UC1);
		float[] pixel_array = new float[neighbors];
		float sum;
		for (int i = radius; i < src.rows() - radius; i++) {
			for (int j = radius; j < src.cols() - radius; j++) {
				float center = (float)src.get(i, j)[0];
				sum = 0;
				for (int n = 0; n < neighbors; n++) {
					float x = i + (float)(radius * Math.cos(2.0 * Math.PI * n / (float)neighbors));
					float y = j - (float)(radius * Math.sin(2.0 * Math.PI * n / (float)neighbors));
					int fx = (int)(Math.floor(x));
					int fy = (int)(Math.floor(y));
					int cx = (int)(Math.ceil(x));
					int cy = (int)(Math.ceil(y));
					float ty = y - fy;
					float tx = x - fx;
					float w1 = (1 - tx) * (1 - ty);
					float w2 = tx * (1 - ty);
					float w3 = (1 - tx) * ty;
					float w4 = tx * ty;
					float t = (float)(w1 * src.get(fx, fy)[0] + w2 * src.get(cx, fy)[0] + w3 * src.get(fx, cy)[0] + w4 * src.get(cx, cy)[0]);
					sum += Math.pow(t - center, 2);
					pixel_array[n] = t;
				}
				float thre = adaption == true ? 0.003f * sum : threshold; //0.003 is the variance threshold
				for (int k = 0; k < neighbors; k++) {
					dst.put(i - radius, j - radius, (int)dst.get(i - radius, j - radius)[0] + Math.pow(2, k) * (pixel_array[k] - src.get(i, j)[0] <= -thre ? 1 : 0));
				}
			}
		}
		return dst;
	}
	
}
