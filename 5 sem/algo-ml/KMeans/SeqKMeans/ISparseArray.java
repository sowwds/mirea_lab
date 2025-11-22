import java.util.*;

interface ISparseArray<T> extends Iterable<IIndexedData<T>> {
    /**
     * Add element to the array at position.
     * 
     * @param position  position in the array
     * @param element   data to place in the array
     */
    void put (int position, T element);

    /**
     * Get element at the given position.
     *
     * @param position  position in the array
     * @return          element at that position or null if there is none
     */
    T get (int position);

    /**
     * Create an iterator over the array.
     *
     * @return  an iterator over the sparse array
     */
    Iterator<IIndexedData<T>> iterator ();
}