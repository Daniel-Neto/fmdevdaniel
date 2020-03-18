import { combineReducers } from 'redux';
import { connectRouter } from 'connected-react-router';

import lms from './lms';
import auth from './auth';
import chart from './chart';
import dialog from './dialog';
import screen from './screen';
import course from './course';
import subject from './subject';
import semester from './semester';
import indicator from './indicator';
import pre_processing from './pre_processing';
import { reducer as toastr } from 'react-redux-toastr';

export default history => combineReducers({
  lms,
  auth,
  chart,
  dialog,
  toastr,
  screen,
  course,
  subject,
  semester,
  indicator,
  pre_processing,
  router: connectRouter(history)
});