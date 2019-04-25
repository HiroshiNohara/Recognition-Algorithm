
public class entrance {
	public static void main(String[] args) {
		traintest traintest = new traintest();
		int trainline = traintest.getlinenum("P:\\train.txt", ';');
		int predictline = traintest.getlinenum("P:\\predict.txt", ';');
		//int Rin = 1, Rex = 4;
		int _grid_x = 1, _grid_y = 1;
		traintest.train("P:\\train.txt", ';', trainline, _grid_x, _grid_y);
		traintest.predict("P:\\predict.txt", ';', predictline, _grid_x, _grid_y);
	}
}
