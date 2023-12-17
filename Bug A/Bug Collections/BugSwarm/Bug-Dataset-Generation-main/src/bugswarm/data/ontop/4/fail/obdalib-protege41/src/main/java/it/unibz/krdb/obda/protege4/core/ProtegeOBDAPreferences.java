package it.unibz.krdb.obda.protege4.core;

/*
 * #%L
 * ontop-protege4
 * %%
 * Copyright (C) 2009 - 2013 KRDB Research Centre. Free University of Bozen Bolzano.
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

import it.unibz.krdb.obda.utils.OBDAPreferences;

import org.protege.editor.core.Disposable;

public class ProtegeOBDAPreferences extends OBDAPreferences 
		implements Disposable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -4772516758684293329L;

	public void dispose() {
		// Do nothing.
	}
}
