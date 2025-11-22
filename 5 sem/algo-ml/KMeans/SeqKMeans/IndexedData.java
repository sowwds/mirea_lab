public class IndexedData<T> implements IIndexedData<T> {
    private int index;
    private T   data;

    public IndexedData (int position, T element) {
        index = position;
        data  = element;
    }

    public int getIndex() {
        return index;
    }

    public T getData() {
        return data;
    }
}
