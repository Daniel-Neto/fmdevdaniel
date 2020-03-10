import { createActions, createReducer } from 'reduxsauce';
import Immutable from 'seamless-immutable';

export const { Types, Creators } = createActions({
  indicatorInit: [],
  indicatorRequest: [],
  indicatorSuccess: ['data'],
  indicatorError: ['err'],
  getIndicators: [],
  setIndicator: ['name', 'value']
});

/** --------------------------------
 * Variable declarations
 * --------------------------------- */

const INITIAL_STATE = Immutable({
  data: [],
  loading: false,
  error: false,
  source: [],
  target: [],
  courseSelected: [],
  subjectSelected: [],
  semesterSelected: []
});

/* Reducers */

export const init = state => state.merge({ data: [] });

export const set = (state, { name, value }) => state.merge({ [name]: value });

export const request = state => state.merge({ loading: true });

export const success = (state, { data }) => state.merge({ source: data, data, error: false, loading: false, target: [] });

export const error = state => state.merge({ loading: false, error: true });

/* Reducers to types */

export default createReducer(INITIAL_STATE, {
  [Types.INDICATOR_INIT]: init,
  [Types.SET_INDICATOR]: set,
  [Types.INDICATOR_REQUEST]: request,
  [Types.INDICATOR_SUCCESS]: success,
  [Types.INDICATOR_ERROR]: error
});