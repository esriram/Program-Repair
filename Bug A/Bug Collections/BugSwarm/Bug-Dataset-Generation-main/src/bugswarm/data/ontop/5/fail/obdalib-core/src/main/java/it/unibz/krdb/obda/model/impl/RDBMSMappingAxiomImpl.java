package it.unibz.krdb.obda.model.impl;

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

import it.unibz.krdb.obda.model.CQIE;
import it.unibz.krdb.obda.model.OBDARDBMappingAxiom;
import it.unibz.krdb.obda.model.OBDASQLQuery;

public class RDBMSMappingAxiomImpl extends AbstractOBDAMappingAxiom implements OBDARDBMappingAxiom {

	private static final long serialVersionUID = 5793656631843898419L;
	
	private OBDASQLQuery sourceQuery;
	private CQIE targetQuery;

	protected RDBMSMappingAxiomImpl(String id, OBDASQLQuery sourceQuery, CQIE targetQuery) {
		super(id);
		setSourceQuery(sourceQuery);
		setTargetQuery(targetQuery);
	}

	@Override
	public void setSourceQuery(OBDASQLQuery query) {
		this.sourceQuery = query;
	}

	@Override
	public void setTargetQuery(CQIE query) {
		this.targetQuery = query;
	}

	@Override
	public OBDASQLQuery getSourceQuery() {
		return sourceQuery;
	}

	@Override
	public CQIE getTargetQuery() {
		return targetQuery;
	}

	@Override
	public OBDARDBMappingAxiom clone() {
		OBDARDBMappingAxiom clone = new RDBMSMappingAxiomImpl(this.getId(), sourceQuery.clone(),targetQuery.clone());
		return clone;
	}
	
	@Override
	public String toString() {
		StringBuffer bf = new StringBuffer();
		bf.append(sourceQuery.toString());
		bf.append(" ==> ");
		bf.append(targetQuery.toString());
		return bf.toString();
	}
}
