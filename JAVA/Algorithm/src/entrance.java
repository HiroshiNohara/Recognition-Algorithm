
public class entrance {
	public static void main(String[] args) {
		traintest traintest = new traintest();
		int trainline = traintest.getlinenum("P:\\train.txt", ';');
		int predictline = traintest.getlinenum("P:\\predict.txt", ';');
		//int Rin = 1, Rex = 4;
		//float threshold = 5.0f;
		//bool adaption = false;
		int _grid_x = 8, _grid_y = 8;
		traintest.train("P:\\train.txt", ';', trainline, _grid_x, _grid_y);
		traintest.predict("P:\\predict.txt", ';', predictline, _grid_x, _grid_y);
	}
}
