import java.io.*;

/**
 * Implementation of the DoubleVector that uses a sparse representation (that is,
 * not all of the entries in the vector are explicitly represented).  All non-default
 * entires in the vector are stored in a HashMap.
 */
class SparseDoubleVector implements IDoubleVector {
  
  // this stores all of the non-default entries in the sparse vector
  private ISparseArray<Double> nonEmptyEntries;
  
  // how many entries there are in the vector
  private int myLength;
 
  /**
   * Creates a new DoubleVector of the specified length
   * 
   * @param len  the length of the vector we are creating
   */
  public SparseDoubleVector (int len) {
    myLength = len;
    nonEmptyEntries = new LinearSparseArray<Double>(3);
  }
  
  public void normalize () {
  
    double total = l1Norm ();
    for (IIndexedData<Double> el : nonEmptyEntries) {
      Double value = el.getData();
      int position = el.getIndex();
      double newValue = value / total;
      nonEmptyEntries.put(position, newValue);
    }
  }
  
  public double l1Norm () {
    
    // iterate through all of the values
    double returnVal = 0.0;
    int nonEmptyEls = 0;
    for (IIndexedData<Double> el : nonEmptyEntries) {
      returnVal += el.getData();
      nonEmptyEls += 1;
    }
    
    return returnVal;
  }
  
  public void setToOne () {
    
    for (IIndexedData<Double> el : nonEmptyEntries) {
      Double value = el.getData();
      int position = el.getIndex();
      if (value > 1e-99) {
        value = 1.0;
        nonEmptyEntries.put(position, value);
      }
    }
        
  }
  
  public void computeLog () {
    
    for (IIndexedData<Double> el : nonEmptyEntries) {
      Double value = el.getData();
      int position = el.getIndex();
      nonEmptyEntries.put(position, Math.log (value));
    }
    
  }
  
  public void subtractMyselfFromHim (IDoubleVector subtractFromHim)  {
    // make sure that the two vectors have the same length
    if (getLength() != subtractFromHim.getLength ()) {
      throw new RuntimeException ("unequal lengths in addMyselfToHim");
    }
  
    // add every non-default value to the other guy
    for (IIndexedData<Double> el : nonEmptyEntries) {
      double myVal = el.getData();
      int curIndex = el.getIndex();
      myVal = subtractFromHim.getItem (curIndex) - myVal;
      subtractFromHim.setItem (curIndex, myVal);
    }
    
  }
    
  public void addMyselfToHim (IDoubleVector addToHim)  {
    // make sure that the two vectors have the same length
    if (getLength() != addToHim.getLength ()) {
      throw new RuntimeException ("unequal lengths in addMyselfToHim");
    }
  
    // add every non-default value to the other guy
    for (IIndexedData<Double> el : nonEmptyEntries) {
      double myVal = el.getData();
      int curIndex = el.getIndex();
      myVal += addToHim.getItem (curIndex);
      addToHim.setItem (curIndex, myVal);
    }
    
  }
  
  public double distance (IDoubleVector fromMe) {
    
    // make sure that the two vectors have the same length
    if (getLength() != fromMe.getLength ()) {
      throw new RuntimeException ("unequal lengths in multiplyIntoHim");
    }
    
    // first, get a copy of fromMe
    SparseDoubleVector temp = new SparseDoubleVector (fromMe.getLength ());
    fromMe.addMyselfToHim (temp);
    
    // and subtract myself from him
    subtractMyselfFromHim (temp);
    
    double res = 0.0;
    for (IIndexedData<Double> el : temp.nonEmptyEntries) {
      double myVal = el.getData();
      res += myVal * myVal;
    }
    return res;
  }
  
  public void multiplyMyselfByHim (IDoubleVector byHim) {
    
    // make sure that the two vectors have the same length
    if (getLength() != byHim.getLength ()) {
      throw new RuntimeException ("unequal lengths in multiplyIntoHim");
    }
                              
    // multiply every non-default value to the other guy
    for (IIndexedData<Double> el : nonEmptyEntries) {
      double myVal = el.getData();
      int curIndex = el.getIndex();
      
      myVal = myVal * byHim.getItem (curIndex);
      nonEmptyEntries.put (curIndex, myVal);
    }
    
  }
  
  public double getItem (int whichOne) {
        
    // make sure we are not out of bounds
    if (whichOne >= myLength) {
      throw new RuntimeException ("index too large in getItem");
    }
    
    // now, look the thing up
    Double myVal = nonEmptyEntries.get (whichOne);
    if (myVal == null) {
      return 0;
    } else {
      return myVal;
    }
  }
  
  public void setItem (int whichOne, double setToMe) {

    // make sure we are not out of bounds
    if (whichOne >= myLength) {
      throw new RuntimeException ("index too large in setItem");
    }
    
    // try to put the value in
    nonEmptyEntries.put (whichOne, setToMe);
  }
  
  public int getLength () {
    return myLength;
  }
  
  public String writeOut () {
  
    ByteArrayOutputStream output = new ByteArrayOutputStream ();
    OutputStreamWriter myOut = new OutputStreamWriter (output);
    try {
      myOut.write ("len: " + myLength + ";");
      for (IIndexedData<Double> el : nonEmptyEntries) {
        double myVal = el.getData();
        int curIndex = el.getIndex();
        myOut.write ( " " + curIndex + ":" + myVal + ";");
      }
      myOut.flush ();
    } catch (Exception e) {
      e.printStackTrace ();
      throw new RuntimeException (e);
    }
    return output.toString ();
  }
  
  public void multiplyMyselfByHim (double byHim) {
    
    for (IIndexedData<Double> el : nonEmptyEntries) {
      Double value = el.getData();
      int position = el.getIndex();
      nonEmptyEntries.put(position, byHim * value);
    }  
  }
  
  public void invert () {
    
    for (IIndexedData<Double> el : nonEmptyEntries) {
      Double value = el.getData();
      int position = el.getIndex();
      nonEmptyEntries.put(position, 1.0 / value);
    }    
  }
  
  // this reads the vector from a string
  public void readIn (String inString) {
    
    // first get the length of the vector
    int endOfLen = inString.indexOf (";");
    String len = inString.substring (5, endOfLen);
    myLength = Integer.parseInt (len);
    
    // and get all of the other guys
    int nextOne = endOfLen + 2;
    nonEmptyEntries = new LinearSparseArray<Double>(3);
    while (true) {
      
      int endOfDim = inString.indexOf (":", nextOne);
      
      // if we did not find another ":" it means we are done
      if (endOfDim == -1)
        break;
      
      int whichDim = Integer.parseInt (inString.substring (nextOne, endOfDim));
      int endOfVal = inString.indexOf (";", nextOne);
      double value = Double.valueOf (inString.substring (endOfDim + 1, endOfVal));
      nonEmptyEntries.put (whichDim, value);
      nextOne = endOfVal + 2;
    }
  }
}
