import blpapi
from pandas import Series
from pandas import DataFrame
import pandas as pd
import datetime
import numpy as np

def getReleaseDateList(request_security, start_date, end_date):
    dateList = []

    session = blpapi.Session()
    if not session.start():
        print("Failed to start session.")
    if not session.openService("//blp/refdata"):
        print("Failed to open service.")

    service = session.getService("//blp/refdata")
    request = service.createRequest("ReferenceDataRequest")

    request.append("securities", request_security)
    request.append("fields", "ECO_RELEASE_DT_LIST")

    overrides = request.getElement("overrides")
    override1 = overrides.appendElement()
    override1.setElement("fieldId", "START_DT")
    override1.setElement("value", start_date)
    override2 = overrides.appendElement()
    override2.setElement("fieldId", "END_DT")
    override2.setElement("value", end_date)

    session.sendRequest(request)

    endReached = False
    while endReached == False:
        ev = session.nextEvent()
        if ev.eventType() == blpapi.Event.RESPONSE or ev.eventType() == blpapi.Event.PARTIAL_RESPONSE:
            for msg in ev:
                fieldData = msg.getElement("securityData").getValue(0).getElement("fieldData").getElement("ECO_RELEASE_DT_LIST")
                for i in range(fieldData.numValues()):
                    dt = fieldData.getValue(i).getElement("Ecostats Release Date").getValue()
                    dateList = np.append(dateList, dt)
        if ev.eventType() == blpapi.Event.RESPONSE:
            endReached = True

    return dateList

class RequestError(Exception):
    """A RequestError is raised when there is a problem with a Bloomberg API response."""

    def __init__(self, value, description):
        self.value = value
        self.description = description

    def __str__(self):
        return self.description + '\n\n' + str(self.value)

class BLPInterface:
    """ A wrapper for the Bloomberg API that returns DataFrames.  This class
        manages a //BLP/refdata service and therefore does not handle event
        subscriptions.

        All calls are blocking and responses are parsed and returned as
        DataFrames where appropriate.

        A RequestError is raised when an invalid security is queried.  Invalid
        fields will fail silently and may result in an empty DataFrame.
    """

    def __init__(self, host='localhost', port=8194, open=True):
        self.active = False
        self.host = host
        self.port = port
        if open:
            self.open()

    def open(self):
        if not self.active:
            sessionOptions = blpapi.SessionOptions()
            sessionOptions.setServerHost(self.host)
            sessionOptions.setServerPort(self.port)
            self.session = blpapi.Session(sessionOptions)
            self.session.start()
            self.session.openService('//BLP/refdata')
            self.refDataService = self.session.getService('//BLP/refdata')
            self.active = True

    def close(self):
        if self.active:
            self.session.stop()
            self.active = False

    def historicalRequest(self, securities, fields, startDate, endDate, **kwargs):
        """ Equivalent to the Excel BDH Function.

            If securities are provided as a list, the returned DataFrame will
            have a MultiIndex.
        """
        defaults = {'startDate': startDate,
                    'endDate': endDate,
                    'periodicityAdjustment': 'ACTUAL',
                    'periodicitySelection': 'DAILY',
                    'nonTradingDayFillOption': 'ACTIVE_DAYS_ONLY',
                    'adjustmentNormal': True,
                    'adjustmentAbnormal': True,
                    'adjustmentSplit': True,
                    'adjustmentFollowDPDF': False}
        defaults.update(kwargs)

        response = self.sendRequest('HistoricalData', securities, fields, defaults)

        data = []
        keys = []

        for msg in response:
            securityData = msg.getElement('securityData')
            fieldData = securityData.getElement('fieldData')
            fieldDataList = [fieldData.getValueAsElement(i) for i in range(fieldData.numValues())]

            df = DataFrame()

            for fld in fieldDataList:
                for v in [fld.getElement(i) for i in range(fld.numElements()) if fld.getElement(i).name() != 'date']:
                    # df.ix[fld.getElementAsDatetime('date'), str(v.name())] = v.getValue()
                    df.loc[fld.getElementAsDatetime('date'), str(v.name())] = v.getValue()

            df.index = pd.to_datetime(df.index)
            df.replace('#N/A History', np.nan, inplace=True)

            keys.append(securityData.getElementAsString('security'))
            data.append(df)

        if len(data) == 0:
            return DataFrame()
        if type(securities) == str:
            data = pd.concat(data, axis=1)
            data.columns.name = 'Field'
        else:
            data = pd.concat(data, keys=keys, axis=1, names=['Security', 'Field'])

        data.index.name = 'Date'
        return data

    def referenceRequest(self, securities, fields, **kwargs):
        """ Equivalent to the Excel BDP Function.

            If either securities or fields are provided as lists, a DataFrame
            will be returned.
        """
        response = self.sendRequest('ReferenceData', securities, fields, kwargs)

        data = DataFrame()

        for msg in response:
            securityData = msg.getElement('securityData')
            securityDataList = [securityData.getValueAsElement(i) for i in range(securityData.numValues())]

            for sec in securityDataList:
                fieldData = sec.getElement('fieldData')
                fieldDataList = [fieldData.getElement(i) for i in range(fieldData.numElements())]

                for fld in fieldDataList:
                    # data.ix[sec.getElementAsString('security'), str(fld.name())] = fld.getValue()
                    data.loc[sec.getElementAsString('security'), str(fld.name())] = fld.getValue()

        if data.empty:
            return data
        else:
            data.index.name = 'Security'
            data.columns.name = 'Field'
            return data.iloc[0, 0] if ((type(securities) == str) and (type(fields) == str)) else data

    def bulkRequest(self, securities, fields, **kwargs):
        """ Equivalent to the Excel BDS Function.

            If securities are provided as a list, the returned DataFrame will
            have a MultiIndex.

            You may pass a list of fields to a bulkRequest.  An appropriate
            Index will be generated, however such a DataFrame is unlikely to
            be useful unless the bulk data fields contain overlapping columns.
        """
        response = self.sendRequest('ReferenceData', securities, fields, kwargs)

        data = []
        keys = []

        for msg in response:
            securityData = msg.getElement('securityData')
            securityDataList = [securityData.getValueAsElement(i) for i in range(securityData.numValues())]

            for sec in securityDataList:
                fieldData = sec.getElement('fieldData')
                fieldDataList = [fieldData.getElement(i) for i in range(fieldData.numElements())]

                df = DataFrame()

                for fld in fieldDataList:
                    for v in [fld.getValueAsElement(i) for i in range(fld.numValues())]:
                        s = Series()
                        for d in [v.getElement(i) for i in range(v.numElements())]:
                            s[str(d.name())] = d.getValue()
                        df = df.append(s, ignore_index=True)

                if not df.empty:
                    keys.append(sec.getElementAsString('security'))
                    data.append(df.set_index(df.columns[0]))

        if len(data) == 0:
            return DataFrame()
        if type(securities) == str:
            data = pd.concat(data, axis=1)
            data.columns.name = 'Field'
        else:
            data = pd.concat(data, keys=keys, axis=0, names=['Security', data[0].index.name])

        return data

    def sendRequest(self, requestType, securities, fields, elements):
        """ Prepares and sends a request then blocks until it can return
            the complete response.

            Depending on the complexity of your request, incomplete and/or
            unrelated messages may be returned as part of the response.
        """
        request = self.refDataService.createRequest(requestType + 'Request')

        if type(securities) == str:
            securities = [securities]
        if type(fields) == str:
            fields = [fields]

        for s in securities:
            request.getElement("securities").appendValue(s)
        for f in fields:
            request.getElement("fields").appendValue(f)
        for k, v in elements.items():
            if type(v) == datetime:
                v = v.strftime('%Y%m%d')
            request.set(k, v)

        self.session.sendRequest(request)

        response = []
        while True:
            event = self.session.nextEvent(100)
            for msg in event:
                if msg.hasElement('responseError'):
                    raise RequestError(msg.getElement('responseError'), 'Response Error')
                if msg.hasElement('securityData'):
                    if msg.getElement('securityData').hasElement('fieldExceptions') and (
                        msg.getElement('securityData').getElement('fieldExceptions').numValues() > 0):
                        raise RequestError(msg.getElement('securityData').getElement('fieldExceptions'), 'Field Error')
                    if msg.getElement('securityData').hasElement('securityError'):
                        raise RequestError(msg.getElement('securityData').getElement('securityError'), 'Security Error')

                if msg.messageType() == requestType + 'Response':
                    response.append(msg)

            if event.eventType() == blpapi.Event.RESPONSE:
                break

        return response

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()

# if __name__ == "__main__":
#     blp = BLPInterface()
#     print blp.referenceRequest('COSTNFR INDEX', 'ECO_RELEASE_DT')
#     print blp.historicalRequest('COSTNFR INDEX', 'ECO_RELEASE_DT','20160101','20161231')
