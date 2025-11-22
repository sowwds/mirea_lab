import java.util.*;

public class LinearIndexedIterator<T> implements Iterator<IIndexedData<T>> {
    private int curIdx;
    private int numElements;
    private int [] indices;
    private Vector<T> data;

    public LinearIndexedIterator (int [] indices, Vector<T> data, int numElements) {
        this.curIdx = -1;
        this.numElements = numElements;
        this.indices = indices;
        this.data = data;
    }

    public boolean hasNext() {
        return (curIdx+1) < numElements;
    }

    public IndexedData<T> next() {
        curIdx++;
        return new IndexedData<T> (indices[curIdx], data.get(curIdx));
    }
    
    public void remove() throws UnsupportedOperationException {
        throw new UnsupportedOperationException();
    }
}