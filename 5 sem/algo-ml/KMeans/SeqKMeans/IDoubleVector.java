
/**
 * This is the interface for a vector of double-precision floating point values.
 */
interface IDoubleVector {
  
  /** 
   * Adds myelf to the contens of another double vector.  
   * Will throw an exception if the two vectors
   * don't have exactly the same sizes.
   * 
   * @param addToHim       the double vector to add to
   */
  public void addMyselfToHim (IDoubleVector addToHim);

  /** 
   * Subtract myelf to the contens of another double vector.  
   * Will throw an exception if the two vectors
   * don't have exactly the same sizes.
   * 
   * @param subtractFromHim       the double vector to subtract from
   */
  public void subtractMyselfFromHim (IDoubleVector subtractFromHim);
    
  /** 
   * Multiplies this double vector by the contents of another one.  
   *  Will throw an exception if the two vectors
   * don't have exactly the same sizes.
   * 
   * @param byHim       the double vector to mult by
   */ 
  public void multiplyMyselfByHim (IDoubleVector byHim);
  
  /** 
   * Multiplies this double vector by a scalar vaue
   * 
   * @param byHim       the double vector to mult by
   */ 
  public void multiplyMyselfByHim (double byHim);
  
  /** 
   * Compute the Euclidean distance from another vector
   * 
   * @param fromHim       the double vector check the distance to
   */ 
  public double distance (IDoubleVector fromHim);
  
  /** 
   * Replaces each entry in the vector with 1/x... so <2, 3, 4> becomes <1/2, 1/3, 1/4>
   */ 
  public void invert ();
 
  /**
   * Returns a particular item in the double vector.   Will throw an exception if the index passed
   * in is beyond the end of the vector.
   * 
   * @param whichOne     the index of the item to return
   * @return             the value at the specified index
   */
  public double getItem (int whichOne);

  /**
   * Sets every non-zero entry in the vector to be 1.0
   */
  public void setToOne ();
  
  /**
   * This forces the sum of all of the items in the vector to be one, by dividing each item by the
   * total over all items.
   */
  public void normalize ();
  
  /**
   * Take the natural logarithm of everything in this vector
   */
  public void computeLog ();
  
  /**
   * Sets a particular item in the vector.  
   *  Will throw an exception if we are trying to set an item
   * that is past the end of the vector.
   * 
   * @param whichOne     the index of the item to set
   * @param setToMe      the value to set the item to
   */
  public void setItem (int whichOne, double setToMe);

  /**
   * Returns the length of the vector.
   * 
   * @return     the vector length
   */
  public int getLength ();
  
  /**
   * Returns the L1 norm of the vector (this is just the sum of the absolute value of all of the entries)
   * 
   * @return      the L1 norm
   */
  public double l1Norm ();
  
}



