
/**
 * This object stores a key-value pair, plus a "location" which is an IDoubleVector 
 * object. This can be used, for example, to store the name of a text document (the key), the
 * label of the document (the value), and a vector that describes the fraction of each of the
 * words in a dictionary that are found in the document.
 */
class VectorizedObject {
 
  // this is the key of the object
  String key;
  
  // this is some data associated with the object
  String value;
  
  // and this is the location of the object
  SparseDoubleVector location;
  
  /**
   * Create and load up a VectorizedObject
   * 
   * @keyIn        the key to store
   * @vlaueIn      the value to store
   * @locationIn   the vector to store
   */
  public VectorizedObject (String keyIn, String valueIn, SparseDoubleVector locationIn) {
    key = keyIn;
    value = valueIn;
    location = locationIn;
  }
  
  /**
   * Constructor that loads this object from a text string.  This is the "inverse" of the
   * writeOut operation, in the sense that if you first write the oject to a text string
   * and then create a new object using this constructor, you'll get back exactly what you
   * started with.
   * 
   * @inString    the String object to read from
   */
  public VectorizedObject (String inString) {
   
    // get the data
    int startOfData = inString.indexOf ("key: ");
    int endOfData = inString.indexOf (";");
    key = inString.substring (startOfData + 5, endOfData);
    
    // get the label
    int startOfLabel = inString.indexOf ("value: ", endOfData);
    int endOfLabel = inString.indexOf (";", startOfLabel);
    value = inString.substring (startOfLabel + 7, endOfLabel);
    
    // and now, get the vector
    location = new SparseDoubleVector (0);
    location.readIn (inString.substring (endOfLabel + 2));
  }
  
  /**
   * Create a copy of the VectorizedObject
   */
  public VectorizedObject copy () {
    String temp = writeOut ();
    return new VectorizedObject (temp);
  }
  
  /**
   * Add a new key into the object, replacing the old one
   * 
   * @keyIn    the key to add
   */
  public void setKey (String keyIn) {
    key = keyIn;  
  }
  
  /**
   * Add a new value into the object, replacing the old one
   * 
   * @valueIn
   */
  public void setValue (String valueIn) {
    value = valueIn;  
  }
  
  /**
   * Treat the value as an integer, and increment it.  This will throw an exception
   * if the value is a string that cannot be interpreted as an integer
   */
  public void incrementValueAsInt () {
    Integer temp = Integer.parseInt (value);
    temp++;
    value = temp.toString (); 
  }
  
  /**
   * Treat the value as an integer, and return it.  This will throw an exception if the
   * value is a string that cannot be interpreted as an integer.
   */
  public int getValueAsInt () {
    return Integer.parseInt (value);
  }
  
  /**
   * Treat the value as an integer, and add to it.  This will throw an exception if the
   * value is a string that cannot be interpreted as an integer.
   */
  public void addToValueAsInt (int addIn) {
    Integer newOne = addIn + + Integer.parseInt (value);
    value = newOne.toString ();
  }
  
  /**
   * Get the current value of the key
   */
  public String getKey () {
    return key; 
  }
  
  /**
   * Get the current value of the value
   */
  public String getValue () {
    return value;
  }
  
  /**
   * Get the current value of the location vector
   */
  public SparseDoubleVector getLocation () {
    return location; 
  }
  
  /**
   * Write a textual version of the object.
   */
  public String writeOut () {
    return ("key: " + key + "; value: " + value + "; ") + location.writeOut (); 
  }
}