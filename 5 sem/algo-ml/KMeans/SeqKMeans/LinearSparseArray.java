import java.util.*;

class LinearSparseArray<T> implements ISparseArray<T> {
    private int [] indices;
    private Vector <T> data;
    private int numElements;
    private int lastSlot;
    private int lastSlotReturned;

    private void doubleCapacity () {
     data.ensureCapacity (lastSlot * 2);
     int [] temp = new int [lastSlot * 2];
     for (int i = 0; i < numElements; i++) {
       temp[i] = indices[i];  
     }
     indices = temp;
     lastSlot *= 2;
    }
    
    public LinearIndexedIterator<T> iterator () {
        return new LinearIndexedIterator<T> (indices, data, numElements);
    }
     
    public LinearSparseArray(int initialSize) {
        indices = new int[initialSize];
        data = new Vector<T> (initialSize);
        numElements = 0;
        lastSlotReturned = -1;
        lastSlot = initialSize;
    }

    public void put (int position, T element) {
      
      // first see if it is OK to just append
      if (numElements == 0 || position > indices[numElements - 1]) {
        data.add (element);
        indices[numElements] = position;
      
      // if the one ew are adding is not the largest, can't just append
      } else {

        // first check to see if we have data at that position
        for (int i = 0; i < numElements; i++) {
          if (indices[i] == position) {
            data.setElementAt (element, i);
            return;
          }
        }
        
        // if we made it here, there is no data at the position
        int pos;
        for (pos = numElements; pos > 0 && indices[pos - 1] > position; pos--) {
          indices[pos] = indices[pos - 1];
        }
        indices[pos] = position;
        data.add (pos, element);
      }
      
      numElements++;
      if (numElements == lastSlot) 
        doubleCapacity ();
    }

    public T get (int position) {

      // first we find an index value that is less than or equal to the position we want
      for (; lastSlotReturned >= 0 && indices[lastSlotReturned] >= position; lastSlotReturned--);
    
      // now we go up, and find the first one that is bigger than what we want
      for (; lastSlotReturned + 1 < numElements && indices[lastSlotReturned + 1] <= position; lastSlotReturned++);
    
      // now there are three cases... if we are at position -1, then we have a very small guy
      if (lastSlotReturned == -1)
        return null;
    
      // if the guy where we are is less than what we want, we didn't find it
      if (indices[lastSlotReturned] != position)
        return null;
    
      // otherwise, we got it!
      return data.get(lastSlotReturned);
    }
}
