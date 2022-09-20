package com.vc.deg;

import java.io.IOException;
import java.nio.file.Path;

public interface DynamicExplorationGraph {
	
	public GraphDesigner designer();
	public GraphNavigator navigator(); // TODO remove
	
	/**
	 * Stores the graph structural data and the feature vectors into a file.
	 * It includes the FeatureSpace, Nodes, Edges, Features, Labels but not 
	 * information about the Design process or Navigation settings.
	 * 
	 * @param file
	 * @throws ClassNotFoundException
	 * @throws IOException
	 */
	public void writeToFile(Path file) throws ClassNotFoundException, IOException;
	
	/**
	 * Search the graph for the best nodes matching the query
	 * 
	 * TODO replace return value with SearchEntry[]
	 * 
	 * @param query
	 * @param k
	 * @return
	 */
	public default SearchResult search(FeatureVector query, int k) {
		return search(query, k, 0.1f);
	}
	
	/**
	 * 
	 * @param query
	 * @param k
	 * @param eps Is similar to a search radius factor 0 means low and 1 means high radius to scan
	 * @return
	 */
	public SearchResult search(FeatureVector query, int k, float eps);
	
	
//	public int[] search(MemoryView query, int top);
//	public Node
	
	/**
     * Create an empty new graph
     * 
     * @param space
     * @return
     */
	public static DynamicExplorationGraph newGraph(FeatureSpace space, int edgesPerNode) {
		return GraphFactory.getDefaultFactory().newGraph(space, edgesPerNode);
	}
	
	/**
	 * Load an existing graph
	 * 
	 * @param file
	 * @return
	 */
	public static DynamicExplorationGraph loadGraph(Path file) throws ClassNotFoundException, IOException {
		return GraphFactory.getDefaultFactory().loadGraph(file);
	}
	
	/**
	 * Load an existing graph
	 * 
	 * @param file
	 * @return
	 */
	public static DynamicExplorationGraph loadGraph(Path file, String componentType) throws ClassNotFoundException, IOException {
		return GraphFactory.getDefaultFactory().loadGraph(file, componentType);
	}
}