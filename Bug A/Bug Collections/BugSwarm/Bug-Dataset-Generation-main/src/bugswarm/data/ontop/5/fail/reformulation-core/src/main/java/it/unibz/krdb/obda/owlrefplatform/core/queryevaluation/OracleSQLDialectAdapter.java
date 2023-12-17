package it.unibz.krdb.obda.owlrefplatform.core.queryevaluation;

/*
 * #%L
 * ontop-reformulation-core
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

import it.unibz.krdb.obda.model.OBDAQueryModifiers;

import java.sql.Types;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;

public class OracleSQLDialectAdapter extends SQL99DialectAdapter {

	public static final int VARIABLE_NAME_MAX_LENGTH = 30;
	/**
	 * If the variable needs to be shortcut, length of the number
	 * introduced.
	 */
	public static final int VARIABLE_NUMBER_LENGTH = 3;

	private static Map<Integer, String> SqlDatatypes;
    private Pattern quotes = Pattern.compile("[\"`\\['].*[\"`\\]']");
	static {
		SqlDatatypes = new HashMap<Integer, String>();
		SqlDatatypes.put(Types.DECIMAL, "NUMBER");
		SqlDatatypes.put(Types.FLOAT, "FLOAT");
		SqlDatatypes.put(Types.CHAR, "CHAR");
		SqlDatatypes.put(Types.VARCHAR, "VARCHAR(4000)");
		SqlDatatypes.put(Types.CLOB, "CLOB");
		SqlDatatypes.put(Types.TIMESTAMP, "TIMESTAMP");
	}


	private String databaseVersion ;

	public OracleSQLDialectAdapter() {
		this.databaseVersion = "";

	}

	public OracleSQLDialectAdapter(String databaseVersion) {
		this.databaseVersion = databaseVersion;

	}


	@Override
	public String sqlSlice(long limit, long offset) {

		String version =databaseVersion.split("\\.")[0];
		try {
			int versionInt = Integer.parseInt(version);

			if (versionInt<12) {
				if (limit == 0) {
					return "WHERE 1 = 0";
				}

				if (limit < 0) {
					if (offset < 0) {
						return "";
					} else {

						return String.format("OFFSET %d ROWS", offset);
					}
				} else {
					if (offset < 0) {
						// If the offset is not specified
						return String.format("OFFSET 0 ROWS\nFETCH NEXT %d ROWS ONLY", limit);
					} else {
						return String.format("OFFSET %d ROWS\nFETCH NEXT %d ROWS ONLY", offset, limit);
					}
				}
			}
			}catch (NumberFormatException nfe){
			//not a number  use new concat

		}
		if (limit >= 0 )
			return String.format("WHERE ROWNUM <= %s", limit);
		else
			return "";

	}

	@Override
	public String sqlOrderByAndSlice(List<OBDAQueryModifiers.OrderCondition> conditions, String viewname, long limit, long offset) {

		return  sqlSlice(limit, offset)  + "\n" + sqlOrderBy(conditions,viewname);

	}

	@Override
	public String SHA1(String str) {
//	  		return String.format("dbms_crypto.HASH(%s, 3)", str);
		return String.format("LOWER(TO_CHAR(RAWTOHEX(SYS.DBMS_CRYPTO.HASH(UTL_I18N.STRING_TO_RAW(%s, 'AL32UTF8'), SYS.DBMS_CRYPTO.HASH_SH1))))", str);
	  	}
	  
	@Override
	public String MD5(String str) {
//		return String.format("dbms_crypto.HASH(%s, 2)", str);
		return String.format("LOWER(TO_CHAR(RAWTOHEX(SYS.DBMS_CRYPTO.HASH(UTL_I18N.STRING_TO_RAW(%s, 'AL32UTF8'), 2))))", str);
//		return String.format("LOWER( RAWTOHEX( UTL_I18N.STRING_TO_RAW(sys.dbms_obfuscation_toolkit.md5(input_string => %s) , 'AL32UTF8')))", str); //works but deprecated
	}

	@Override
	public String strEndsOperator(){
		return "SUBSTR(%1$s, -LENGTH(%2$s) ) LIKE %2$s";
	}

	@Override
	public String strStartsOperator(){
		return "SUBSTR(%1$s, 0, LENGTH(%2$s)) LIKE %2$s";
	}

	@Override
	public String strContainsOperator(){
		return "INSTR(%1$s,%2$s) > 0";
	}

	@Override
	public String strBefore(String str, String before) {
		return String.format("NVL(SUBSTR(%s, 0, INSTR(%s,%s )-1), '') ", str,  str , before);
	};

	@Override
	public String strAfter(String str, String after) {
		return String.format("NVL(SUBSTR(%s,INSTR(%s,%s)+LENGTH(%s), SIGN(INSTR(%s,%s))*LENGTH(%s)), '')",
				str, str, after, after, str, after, str); //FIXME when no match found should return empty string
	}
//	"SUBSTRING(%s,CHARINDEX(%s,%s)+LEN(%s),SIGN(CHARINDEX(%s,%s))*LEN(%s))",
//	str, after, str , after , after, str, str

	@Override
	public String sqlCast(String value, int type) {
		String strType = SqlDatatypes.get(type);
		if (strType != null) {	
			return "CAST(" + value + " AS " + strType + ")";
		}
		throw new RuntimeException("Unsupported SQL type");
	}
	
	@Override
	public String sqlRegex(String columnname, String pattern, boolean caseinSensitive, boolean multiLine, boolean dotAllMode) {

        if(quotes.matcher(pattern).matches() ) {
            pattern = pattern.substring(1, pattern.length() - 1); // remove the
            // enclosing
            // quotes
        }
		String flags = "";
		if(caseinSensitive)
			flags += "i";
		else
			flags += "c";
		if (multiLine)
			flags += "m";
		if(dotAllMode)
			flags += "n";
		
		String sql = " REGEXP_LIKE " + "( " + columnname + " , '" + pattern + "' , '" + flags  + "' )";
		return sql;
	}

    @Override
    public String strReplace(String str, String oldstr, String newstr) {
        if(quotes.matcher(oldstr).matches() ) {
            oldstr = oldstr.substring(1, oldstr.length() - 1); // remove the enclosing quotes
        }

        if(quotes.matcher(newstr).matches() ) {
            newstr = newstr.substring(1, newstr.length() - 1);
        }
        return String.format("REGEXP_REPLACE(%s, '%s', '%s')", str, oldstr, newstr);
    }

	@Override
	public String dateNow() {
		return "CURRENT_TIMESTAMP";

	}

	@Override
	public String strUuid() {
		return "regexp_replace(rawtohex(sys_guid()), '([A-F0-9]{8})([A-F0-9]{4})([A-F0-9]{4})([A-F0-9]{4})([A-F0-9]{12})', '\\1-\\2-\\3-\\4-\\5')";
	}

	@Override
	public String uuid() {
		return strConcat(new String[] {"'urn:uuid:'", "regexp_replace(rawtohex(sys_guid()), '([A-F0-9]{8})([A-F0-9]{4})([A-F0-9]{4})([A-F0-9]{4})([A-F0-9]{12})', '\\1-\\2-\\3-\\4-\\5')"});
	}

	@Override
	public String rand() {
		return "dbms_random.random";
	}

	@Override
	public String dateTZ(String str) {
		return strConcat(new String[] {String.format("EXTRACT(TIMEZONE_HOUR FROM %s)", str), "':'" , String.format("EXTRACT(TIMEZONE_MINUTE FROM %s) ",str)});
	}

	@Override
	public String getDummyTable() {
		return "SELECT 1 from dual";
	}
	
	@Override 
	public String getSQLLexicalFormBoolean(boolean value) {
		return value ? 	"1" : "0";
	}
	
	/***
	 * Given an XSD dateTime this method will generate a SQL TIMESTAMP value.
	 * The method will strip any fractional seconds found in the date time
	 * (since we haven't found a nice way to support them in all databases). It
	 * will also normalize the use of Z to the timezome +00:00 and last, if the
	 * database is H2, it will remove all timezone information, since this is
	 * not supported there.
	 */
	@Override
	public String getSQLLexicalFormDatetime(String v) {
		String datetime = v.replace('T', ' ');
		int dotlocation = datetime.indexOf('.');
		int zlocation = datetime.indexOf('Z');
		int minuslocation = datetime.indexOf('-', 10); // added search from 10th pos, because we need to ignore minuses in date
		int pluslocation = datetime.indexOf('+');
		StringBuilder bf = new StringBuilder(datetime);
		if (zlocation != -1) {
			/*
			 * replacing Z by +00:00
			 */
			bf.replace(zlocation, bf.length(), "+00:00");
		}

		if (dotlocation != -1) {
			/*
			 * Stripping the string from the presicion that is not supported by
			 * SQL timestamps.
			 */
			// TODO we need to check which databases support fractional
			// sections (e.g., oracle,db2, postgres)
			// so that when supported, we use it.
			int endlocation = Math.max(zlocation, Math.max(minuslocation, pluslocation));
			if (endlocation == -1) {
				endlocation = datetime.length();
			}
			bf.replace(dotlocation, endlocation, "");
		}
		if (bf.length() > 19) {
			bf.delete(19, bf.length());
		}
		bf.insert(0, "'");
		bf.append("'");
		
		/*
		 * Oracle has a special treatment for datetime datatype such that it requires a default
		 * datetime format. In this case, the default is 'YYYY-MM-DD HH24:MI:SS.FF' as in SPARQL
		 * standard, e.g., to_date('2012-12-18 09:58:23.2','YYYY-MM-DD HH24:MI:SS.FF')
		 */
		bf.insert(0, "to_timestamp(");
		bf.append(",'YYYY-MM-DD HH24:MI:SS')");
			
		return bf.toString();
	}

	@Override
	public String getSQLLexicalFormDatetimeStamp(String v) {
		String datetime = v.replace('T', ' ');
		int dotlocation = datetime.indexOf('.');
		int zlocation = datetime.indexOf('Z');
		int minuslocation = datetime.indexOf('-', 10); // added search from 10th pos, because we need to ignore minuses in date
		int pluslocation = datetime.indexOf('+');
		StringBuilder bf = new StringBuilder(datetime);
		if (zlocation != -1) {
			/*
			 * replacing Z by +00:00
			 */
			bf.replace(zlocation, bf.length(), "+00:00");
		}

		if (dotlocation != -1) {
			/*
			 * Stripping the string from the presicion that is not supported by
			 * SQL timestamps.
			 */
			// TODO we need to check which databases support fractional
			// sections (e.g., oracle,db2, postgres)
			// so that when supported, we use it.
			int endlocation = Math.max(zlocation, Math.max(minuslocation, pluslocation));
			if (endlocation == -1) {
				endlocation = datetime.length();
			}
			bf.replace(dotlocation, endlocation, "");
		}
		bf.insert(0, "'");
		bf.append("'");

		/*
		 * Oracle has a special treatment for datetime datatype such that it requires a default
		 * datetime format. To have time zone as required by datetimestamp we can use to_timestamp_tz
		 */
		bf.insert(0, "to_timestamp_tz(");
		bf.append(",'YYYY-MM-DD HH24:MI:SSTZH:TZM')");

		return bf.toString();
	}

	@Override
	public String nameTopVariable(String signatureVariableName, String suffix, Set<String> sqlVariableNames) {
		int suffixLength = suffix.length();
		int signatureVarLength = signatureVariableName.length();

		if (suffixLength >= (VARIABLE_NAME_MAX_LENGTH - VARIABLE_NUMBER_LENGTH))  {
			throw new IllegalArgumentException("The suffix is too long (must be less than " +
					(VARIABLE_NAME_MAX_LENGTH - VARIABLE_NUMBER_LENGTH) + ")");
		}

		/**
		 * If the length limit is not reached, processes as usual.
		 */
		if (signatureVarLength + suffixLength <= VARIABLE_NAME_MAX_LENGTH) {
			return super.nameTopVariable(signatureVariableName, suffix, sqlVariableNames);
		}

		String varPrefix = signatureVariableName.substring(0, VARIABLE_NAME_MAX_LENGTH - suffixLength
				- VARIABLE_NUMBER_LENGTH);


		/**
		 * Naive implementation
		 */
		for (int i = 0; i < Math.pow(10, VARIABLE_NUMBER_LENGTH); i++) {
			String mainVarName = super.nameTopVariable(varPrefix + i, suffix, sqlVariableNames);
			if (!sqlVariableNames.contains(mainVarName)) {
				return mainVarName;
			}
		}

		// TODO: find a better exception
		throw new RuntimeException("Impossible to create a new variable " + varPrefix + "???" + suffix + " : already " +
				Math.pow(10, VARIABLE_NUMBER_LENGTH) + " of them.");
	}
}
