package it.unibz.krdb.obda.utils;

/*
 * #%L
 * ontop-obdalib-core
 * %%
 * Copyright (C) 2009 - 2014 Free University of Bozen-Bolzano
 * %%
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *      http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * #L%
 */

import it.unibz.krdb.obda.exception.DuplicateMappingException;
import it.unibz.krdb.obda.model.CQIE;
import it.unibz.krdb.obda.model.Function;
import it.unibz.krdb.obda.model.OBDADataFactory;
import it.unibz.krdb.obda.model.OBDAMappingAxiom;
import it.unibz.krdb.obda.model.OBDAModel;
import it.unibz.krdb.obda.model.OBDARDBMappingAxiom;
import it.unibz.krdb.obda.model.OBDASQLQuery;
import it.unibz.krdb.obda.model.impl.OBDADataFactoryImpl;

import java.net.URI;
import java.util.ArrayList;
import java.util.List;

/**
 * This class split the mappings
 * 
 *  <pre> q1, q2, ... qn <- SQL </pre>
 *  
 *   into n mappings
 *   
 *  <pre> q1 <-SQL , ..., qn <- SQL </pre>
 * 
 * 
 * @author xiao
 *
 */
public class MappingSplitter {

	private static List<OBDAMappingAxiom> splitMappings(List<OBDAMappingAxiom> mappings) {

		List<OBDAMappingAxiom> newMappings = new ArrayList<OBDAMappingAxiom>();
		
		OBDADataFactory dfac = OBDADataFactoryImpl.getInstance();

		for (OBDAMappingAxiom mapping : mappings) {

			String id = mapping.getId();

			CQIE targetQuery = (CQIE) mapping.getTargetQuery();

			OBDASQLQuery sourceQuery = (OBDASQLQuery) mapping.getSourceQuery();

			Function head = targetQuery.getHead();
			List<Function> bodyAtoms = targetQuery.getBody();

			if(bodyAtoms.size() == 1){
				// For mappings with only one body atom, we do not need to change it
				newMappings.add(mapping);
			} else {
				for (Function bodyAtom : bodyAtoms) {
					String newId = IDGenerator.getNextUniqueID(id + "#");
					
					CQIE newTargetQuery = dfac.getCQIE(head, bodyAtom);
					OBDARDBMappingAxiom newMapping = dfac.getRDBMSMappingAxiom(newId, sourceQuery, newTargetQuery);
					newMappings.add(newMapping);
				}
			}

		}

		return newMappings;

	}

	/**
	 * this method split the mappings in {@link obdaModel} with sourceURI 
	 * 
	 * @param obdaModel
	 * @param sourceURI
	 */
	public static void splitMappings(OBDAModel obdaModel, URI sourceURI) {
		List<OBDAMappingAxiom> splittedMappings = splitMappings(obdaModel.getMappings(sourceURI));
		
		obdaModel.removeAllMappings();
		for(OBDAMappingAxiom mapping : splittedMappings){
			try {
				obdaModel.addMapping(sourceURI, mapping);
			} catch (DuplicateMappingException e) {
				throw new RuntimeException("Error: Duplicate Mappings generated by the MappingSplitter");
			}
		}
				
	}
}
