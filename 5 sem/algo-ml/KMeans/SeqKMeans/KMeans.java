import java.util.*;
import java.io.*;

class KMeans {

  public static void runAnIteration (String dataFile, String clusterFile) {
    
    // start out by opening up the two input files
    BufferedReader inData, inClusters;
    try {
      inData = new BufferedReader (new FileReader (dataFile));
      inClusters = new BufferedReader (new FileReader (clusterFile)); 
    } catch (Exception e) {
      throw new RuntimeException (e); 
    }
    
    // this is the list of current clusters
    ArrayList <VectorizedObject> oldClusters = new ArrayList <VectorizedObject> (0);
    
    // this is the list of clusters that come out of the current iteration
    ArrayList <VectorizedObject> newClusters = new ArrayList <VectorizedObject> (0);
    
    try {     
      
      // read in each cluster... each cluster is on one line of the clusters file
      String cur = inClusters.readLine (); 
      while (cur != null) {
        
        // first, read in the old cluster
        VectorizedObject cluster = new VectorizedObject (cur);
        oldClusters.add (cluster);
        
        // now create the new cluster with the same name (key) as the old one, with zero points
        // assigned, and with a vector at the origin
        VectorizedObject newCluster = new VectorizedObject (cluster.getKey (), "0", 
                                           new SparseDoubleVector (cluster.getLocation ().getLength ())); 
        newClusters.add (newCluster);
        
        // read in the next line
        cur = inClusters.readLine (); 
      }
      inClusters.close ();
      
    } catch (Exception e) {
      throw new RuntimeException (e); 
    }

    // now, process the data
    try {     
      
      // read in each data point... each point is on one line of the file
      String cur = inData.readLine (); 

      // while we have not hit the end of the file
      while (cur != null) {
        
	/*****************************************************************************/
        // Add your code here!
        //
        // It should go through each of the centroids, and find the one closest to the
        // data point stored in "cur". Then, it should add the point that is stored in
	// "cur" into the appropriate, corresponding entry in "newClusters"
	/*****************************************************************************/

        // read in the next line from the file
        cur = inData.readLine (); 
      }
      
      System.out.println ("Done with pass thru data.");
      inData.close ();
      
    } catch (Exception e) {
      e.printStackTrace ();
      throw new RuntimeException (e); 
    }

    // next we will deal with splitting up any of the clusters that have too many points

    /*********************************************************************************/
    // Add your code here!
    //  
    // It should go through all of the clusters, finding the one with the most points,
    // and the one with the fewest.  If the one with the most points has more than 20X
    // the one with the fewest points, replace the centroid of the one with the fewest
    // with the centoid of the one with the most, then multiply the centroid by 1.000001
    // so that they are not the same.
    /*********************************************************************************/

    // lastly, divide each cluster by its count and write it out
    PrintWriter out;
    try {
      out = new PrintWriter (new BufferedWriter (new FileWriter (clusterFile)));
    } catch (Exception e) {
      throw new RuntimeException (e); 
    }
    
    // loops through all of the clusters and writes them out
    for (VectorizedObject i : newClusters) {
      i.getLocation ().multiplyMyselfByHim (1.0 / i.getValueAsInt ());
      String stringRep = i.writeOut ();
      out.println (stringRep);
    }
    
    // and get outta here!
    out.close ();
  }
  
  public static void main (String [] args) {
    
    BufferedReader myConsole = new BufferedReader(new InputStreamReader(System.in));
    if (myConsole == null) {
      throw new RuntimeException ("No console.");
    }
    
    String dataFile = null;
    System.out.format ("Enter the file with the data vectors: ");
    while (dataFile == null) {
      try {
        dataFile = myConsole.readLine ();
      } catch (Exception e) {
        System.out.println ("Problem reading data file name.");
      }
    }
    
    String clusterFile = null;
    System.out.format ("Enter the name of the file where the clusers are loated: ");
    while (clusterFile == null) {
      try {
        clusterFile = myConsole.readLine ();
      } catch (Exception e) {
        System.out.println ("Problem reading file name.");
      }
    }
    
    Integer k = null;
    System.out.format ("Enter the number of iterations to run: ");
    while (k == null) {
      try {
        String temp = myConsole.readLine ();
        k = Integer.parseInt (temp);
      } catch (Exception e) {
        System.out.println ("Problem reading the number of clusters.");
      }
    } 
    
    for (int i = 0; i < k; i++)
      runAnIteration (dataFile, clusterFile); 
  }
}

